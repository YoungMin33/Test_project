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

function emptyAxis() {
  return {
    score: 0,
    confidence: 0,
    evidence: [],
    reason: "No validated evidence."
  };
}

function normalizeText(value: string) {
  return value.replace(/\s+/g, " ").trim().toLowerCase();
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
      evaluation.axes[axis] = emptyAxis();
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

export async function evaluateAnswer({ mission, answer }: EvaluateAnswerInput): Promise<EvaluateAnswerResult> {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is not set.");
  }

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
  const evaluation = validateAndRecalculate(answer, parsed);
  const missionScore = toMissionScore(evaluation, mission);

  return { evaluation, missionScore };
}
