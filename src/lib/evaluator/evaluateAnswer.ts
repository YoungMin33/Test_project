import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { AXES, Evaluation, EvaluationSchema, MissionScore } from "./schema.js";
import { buildEvaluationPrompt, PROMPT_VERSION, SYSTEM_PROMPT } from "./prompts.js";

type Mission = {
  mission_id: string;
  job_name?: string;
  title?: string;
  scenario?: string;
  task?: string;
  axis_signals?: Partial<Record<(typeof AXES)[number], number>>;
  rubric?: Record<string, string[]>;
};

type EvaluateAnswerInput = {
  mission: Mission;
  answer: string;
};

type EvaluateAnswerResult = {
  evaluation: Evaluation;
  missionScore: MissionScore;
};

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

function emptyAxis(reason?: string) {
  return {
    score: 0,
    confidence: 0,
    evidence: [],
    reason: reason ?? "답변에서 해당 축과 관련된 구체적인 행동이나 근거를 찾을 수 없습니다."
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function normalizeText(value: string) {
  return value.replace(/\s+/g, " ").trim().toLowerCase();
}

function countNumbers(value: string) {
  return value.match(/\d+(?:[.,]\d+)?%?|\d+(?:[.,]\d+)?/g)?.length ?? 0;
}

function uniqueTokenRatio(value: string) {
  const tokens = normalizeText(value).split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return 0;
  return new Set(tokens).size / tokens.length;
}

function countMarkers(answer: string, markers: string[]) {
  const normalizedAnswer = normalizeText(answer);
  return markers.filter((marker) => normalizedAnswer.includes(normalizeText(marker))).length;
}

function keywordHits(answer: string, keywords: string[]) {
  const normalizedAnswer = normalizeText(answer);
  return keywords
    .map((keyword) => keyword.trim())
    .filter((keyword) => keyword && normalizedAnswer.includes(normalizeText(keyword)));
}

function quoteAppearsInAnswerStrict(answer: string, quote: string) {
  const normalizedAnswer = normalizeText(answer);
  const normalizedQuote = normalizeText(quote);
  if (!normalizedQuote) return false;
  return normalizedAnswer.includes(normalizedQuote);
}

function addFlag(evaluation: Evaluation, flag: Evaluation["flags"][number]) {
  if (!evaluation.flags.includes(flag)) evaluation.flags.push(flag);
}

function validateAndRecalculate(answer: string, evaluation: Evaluation): Evaluation {
  const usedQuotes = new Set<string>();

  for (const axis of AXES) {
    const axisResult = evaluation.axes[axis];
    axisResult.evidence = axisResult.evidence.filter((item) => {
      const quoteKey = normalizeText(item.quote);
      const supportsAxis = item.primary_axis === axis;
      const isStrictAnswerQuote = quoteAppearsInAnswerStrict(answer, item.quote);
      const isDuplicate = usedQuotes.has(quoteKey);

      if (supportsAxis && isStrictAnswerQuote && !isDuplicate) {
        usedQuotes.add(quoteKey);
        return true;
      }

      return false;
    });

    if (axisResult.evidence.length === 0) {
      const originalReason = axisResult.reason;
      const hasLLMReason = originalReason && originalReason !== "No validated evidence." && originalReason.length > 5;
      evaluation.axes[axis] = emptyAxis(
        hasLLMReason ? `[검증 후 근거 불인정] ${originalReason}` : undefined
      );
      continue;
    }

    const primaryEvidence = axisResult.evidence.filter((item) => item.primary_axis === axis);
    const scoreEvidence = primaryEvidence.length > 0 ? primaryEvidence : axisResult.evidence;
    axisResult.score = Math.max(...scoreEvidence.map((item) => item.level));

    if (primaryEvidence.length === 0) {
      axisResult.score = Math.min(axisResult.score, 2);
      axisResult.reason = `${axisResult.reason} Secondary-only evidence; score capped.`;
    }

    if (axisResult.evidence.length < 2 && axisResult.score >= 4) {
      axisResult.score = 3;
      axisResult.reason = `${axisResult.reason} High score capped because evidence is limited.`;
    }

    axisResult.confidence = Math.min(Math.max(axisResult.confidence, 0), 1);

    if (axisResult.confidence < 0.55) {
      axisResult.score = Math.min(axisResult.score, 2);
    }
  }

  const highAxes = AXES
    .filter((axis) => evaluation.axes[axis].score >= 3)
    .sort((a, b) => {
      const ea = evaluation.axes[a];
      const eb = evaluation.axes[b];
      return (eb.confidence * 10 + eb.evidence.length) - (ea.confidence * 10 + ea.evidence.length);
    });

  if (highAxes.length >= 4) {
    highAxes.slice(2).forEach((axis) => {
      evaluation.axes[axis].score = Math.min(evaluation.axes[axis].score, 2);
      evaluation.axes[axis].confidence = Math.min(evaluation.axes[axis].confidence, 0.54);
      evaluation.axes[axis].reason = `${evaluation.axes[axis].reason} Capped to avoid all-axis high scoring.`;
    });
    addFlag(evaluation, "low_confidence");
  }

  const lowConfidence = AXES.some((axis) => {
    const result = evaluation.axes[axis];
    return result.score > 0 && (result.confidence < 0.55 || result.evidence.length === 0);
  });

  if (lowConfidence) addFlag(evaluation, "low_confidence");

  return evaluation;
}

function toMissionScore(evaluation: Evaluation, mission: Mission): MissionScore {
  return Object.fromEntries(AXES.map((axis) => {
    const signal = mission.axis_signals?.[axis] ?? 0;
    return [axis, (evaluation.axes[axis].score / 4) * signal];
  })) as MissionScore;
}

function heuristicFallbackEvaluation(mission: Mission, answer: string, cause: string): Evaluation {
  const length = normalizeText(answer).length;
  const numberCount = countNumbers(answer);
  const uniqueness = uniqueTokenRatio(answer);
  const repetitionPenalty = uniqueness < 0.45 ? 0.9 : uniqueness < 0.6 ? 0.35 : 0;
  const lengthLevel = length >= 260 ? 3 : length >= 140 ? 2 : length >= 45 ? 1 : 0;
  const markersByAxis: Record<(typeof AXES)[number], string[]> = {
    AX1: ["data", "log", "metric", "rate", "ratio", "compare", "p=", "%"],
    AX2: ["check", "observe", "pattern", "trace", "inspect", "explore"],
    AX3: ["priority", "strategy", "hypothesis", "verify", "decide", "option", "recommend"],
    AX4: ["team", "share", "report", "role", "approve", "department"],
    AX5: ["customer", "user", "empathy", "apologize", "communicate", "trust"]
  };

  const axes = Object.fromEntries(AXES.map((axis) => {
    const hits = keywordHits(answer, mission.rubric?.[axis] ?? []);
    const markerHits = countMarkers(answer, markersByAxis[axis]);
    const signal = mission.axis_signals?.[axis] ?? 0;
    const keywordLevel = Math.min(hits.length, 5);
    const markerLevel = Math.min(markerHits, 4);
    const numericLevel = axis === "AX1" ? Math.min(numberCount, 4) : 0;
    const signalLevel = signal * 4;
    const evidenceShapeBonus = keywordLevel >= 2 && (markerLevel >= 1 || numericLevel >= 1) ? 0.55 : 0;
    const raw =
      keywordLevel * 0.55 +
      markerLevel * 0.35 +
      numericLevel * 0.28 +
      signalLevel * 0.38 +
      lengthLevel * 0.22 +
      evidenceShapeBonus -
      repetitionPenalty;

    let score = 0;
    if (raw >= 4.2 && length >= 120 && uniqueness >= 0.55) score = 4;
    else if (raw >= 3.0 && length >= 80 && uniqueness >= 0.5) score = 3;
    else if (raw >= 1.65 && length >= 35 && uniqueness >= 0.45) score = 2;
    else if (raw >= 0.45) score = 1;

    if (hits.length === 0 && markerHits === 0 && numericLevel === 0) {
      score = 0;
    }

    const confidence = score === 0
      ? clamp(0.08 + signal * 0.1, 0.05, 0.2)
      : clamp(
          0.16 +
          score * 0.08 +
          Math.min(hits.length, 4) * 0.035 +
          Math.min(markerHits, 3) * 0.025 +
          signal * 0.14 +
          lengthLevel * 0.025 -
          repetitionPenalty * 0.08,
          0.18,
          0.68
        );

    return [axis, {
      score,
      confidence: Number(confidence.toFixed(2)),
      evidence: [],
      reason: `Heuristic fallback (${cause}). raw=${raw.toFixed(2)}, keyword_hits=${hits.length}, marker_hits=${markerHits}, signal=${signal.toFixed(2)}, length_level=${lengthLevel}, uniqueness=${uniqueness.toFixed(2)}.`
    }];
  })) as Evaluation["axes"];

  const flags: Evaluation["flags"] = ["low_confidence"];
  if (length < 15) flags.push("too_short");
  if (uniqueTokenRatio(answer) < 0.45) flags.push("ambiguous");

  return {
    mission_id: mission.mission_id,
    axes,
    flags,
    prompt_version: `${PROMPT_VERSION}-fallback`
  };
}

export async function evaluateAnswer({ mission, answer }: EvaluateAnswerInput): Promise<EvaluateAnswerResult> {
  if (!process.env.OPENAI_API_KEY) {
    const evaluation = heuristicFallbackEvaluation(mission, answer, "missing_api_key");
    return { evaluation, missionScore: toMissionScore(evaluation, mission) };
  }

  let evaluation: Evaluation;
  try {
    const response = await openai.responses.parse({
      model: process.env.OPENAI_EVAL_MODEL ?? "gpt-4.1-mini",
      temperature: 0,
      input: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: buildEvaluationPrompt({ mission, answer }) }
      ],
      text: {
        format: zodTextFormat(EvaluationSchema, "jobsim_evaluation")
      }
    });

    const parsed = response.output_parsed;
    if (!parsed) {
      throw new Error("OpenAI returned an empty evaluation.");
    }

    parsed.prompt_version = parsed.prompt_version || PROMPT_VERSION;
    evaluation = validateAndRecalculate(answer, parsed);
  } catch (error) {
    const cause = error instanceof Error ? error.message : "unknown_llm_error";
    evaluation = heuristicFallbackEvaluation(mission, answer, cause);
  }

  const missionScore = toMissionScore(evaluation, mission);

  return { evaluation, missionScore };
}
