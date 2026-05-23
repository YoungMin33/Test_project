import "dotenv/config";
import fs from "node:fs/promises";
import path from "node:path";
import { evaluateAnswer } from "../src/lib/evaluator/evaluateAnswer.js";
import { PROMPT_VERSION, RUBRIC_VERSION } from "../src/lib/evaluator/prompts.js";
import { AXES, type Axis, type Evaluation } from "../src/lib/evaluator/schema.js";

type AxisRange = { min: number; max: number };

type GoldenCase = {
  id: string;
  mission_id: string;
  answer: string;
  expected: {
    axes: Record<Axis, AxisRange>;
    confidenceMin?: number;
    confidenceMax?: number;
    allowFlags?: string[];
    denyFlags?: string[];
    maxScoreRange?: number;
    maxConfidenceRange?: number;
  };
  tags?: string[];
};

type GoldenSuite = {
  version: string;
  defaults?: {
    runs?: number;
    maxScoreRange?: number;
    maxConfidenceRange?: number;
    failHighScoreWithoutEvidenceAt?: number;
  };
  axisBias?: {
    warnMeanAbove?: number;
    warnMeanDeltaAbove?: number;
    watchAxes?: Axis[];
  };
  cases: GoldenCase[];
};

type Mission = {
  mission_id: string;
  job_code: string;
  job_name?: string;
  title?: string;
  scenario?: string;
  task?: string;
  axis_signals?: Partial<Record<Axis, number>>;
  rubric?: Record<string, string[]>;
};

type CaseRun = {
  run: number;
  scores: Record<Axis, number>;
  confidences: Record<Axis, number>;
  evidenceCounts: Record<Axis, number>;
  flags: string[];
  prompt_version: string;
};

type CaseReport = {
  id: string;
  mission_id: string;
  tags: string[];
  status: "pass" | "fail";
  failures: string[];
  drift: {
    maxScoreRange: number;
    maxConfidenceRange: number;
    scoreRangeByAxis: Record<Axis, number>;
    confidenceRangeByAxis: Record<Axis, number>;
  };
  runs: CaseRun[];
};

const root = process.cwd();

function argValue(name: string) {
  const prefix = `--${name}=`;
  return process.argv.find((arg) => arg.startsWith(prefix))?.slice(prefix.length);
}

function hasArg(name: string) {
  return process.argv.includes(`--${name}`);
}

async function readJson<T>(filePath: string): Promise<T> {
  return JSON.parse(await fs.readFile(filePath, "utf8")) as T;
}

function avg(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function minMaxRange(values: number[]) {
  if (!values.length) return 0;
  return Math.max(...values) - Math.min(...values);
}

function scoresFromEvaluation(evaluation: Evaluation): Record<Axis, number> {
  return Object.fromEntries(AXES.map((axis) => [axis, evaluation.axes[axis].score])) as Record<Axis, number>;
}

function confidencesFromEvaluation(evaluation: Evaluation): Record<Axis, number> {
  return Object.fromEntries(AXES.map((axis) => [axis, evaluation.axes[axis].confidence])) as Record<Axis, number>;
}

function evidenceCountsFromEvaluation(evaluation: Evaluation): Record<Axis, number> {
  return Object.fromEntries(AXES.map((axis) => [axis, evaluation.axes[axis].evidence.length])) as Record<Axis, number>;
}

function checkCase(
  testCase: GoldenCase,
  runs: CaseRun[],
  suite: GoldenSuite
) {
  const failures: string[] = [];
  const latest = runs[runs.length - 1];
  const maxScoreRange = testCase.expected.maxScoreRange ?? suite.defaults?.maxScoreRange ?? 1;
  const maxConfidenceRange = testCase.expected.maxConfidenceRange ?? suite.defaults?.maxConfidenceRange ?? 0.35;
  const highWithoutEvidenceAt = suite.defaults?.failHighScoreWithoutEvidenceAt ?? 3;

  const scoreRangeByAxis = Object.fromEntries(AXES.map((axis) => {
    const values = runs.map((run) => run.scores[axis]);
    return [axis, minMaxRange(values)];
  })) as Record<Axis, number>;

  const confidenceRangeByAxis = Object.fromEntries(AXES.map((axis) => {
    const values = runs.map((run) => run.confidences[axis]);
    return [axis, Number(minMaxRange(values).toFixed(4))];
  })) as Record<Axis, number>;

  for (const axis of AXES) {
    const expected = testCase.expected.axes[axis];
    const value = latest.scores[axis];
    if (value < expected.min || value > expected.max) {
      failures.push(`${axis} expected ${expected.min}-${expected.max}, got ${value}`);
    }

    if (scoreRangeByAxis[axis] > maxScoreRange) {
      failures.push(`${axis} score drift ${scoreRangeByAxis[axis]} exceeds ${maxScoreRange}`);
    }

    if (confidenceRangeByAxis[axis] > maxConfidenceRange) {
      failures.push(`${axis} confidence drift ${confidenceRangeByAxis[axis]} exceeds ${maxConfidenceRange}`);
    }

    for (const run of runs) {
      if (run.scores[axis] >= highWithoutEvidenceAt && run.evidenceCounts[axis] === 0) {
        failures.push(`${axis} run ${run.run} scored ${run.scores[axis]} without evidence`);
      }
    }
  }

  const allConfidences = runs.flatMap((run) => AXES.map((axis) => run.confidences[axis]));
  if (testCase.expected.confidenceMin !== undefined && avg(allConfidences) < testCase.expected.confidenceMin) {
    failures.push(`average confidence expected >=${testCase.expected.confidenceMin}, got ${avg(allConfidences).toFixed(3)}`);
  }
  if (testCase.expected.confidenceMax !== undefined && avg(allConfidences) > testCase.expected.confidenceMax) {
    failures.push(`average confidence expected <=${testCase.expected.confidenceMax}, got ${avg(allConfidences).toFixed(3)}`);
  }

  for (const denied of testCase.expected.denyFlags ?? []) {
    if (latest.flags.includes(denied)) failures.push(`denied flag present: ${denied}`);
  }

  return {
    failures,
    drift: {
      maxScoreRange: Math.max(...AXES.map((axis) => scoreRangeByAxis[axis])),
      maxConfidenceRange: Math.max(...AXES.map((axis) => confidenceRangeByAxis[axis])),
      scoreRangeByAxis,
      confidenceRangeByAxis
    }
  };
}

function summarizeAxis(caseReports: CaseReport[]) {
  const finalRuns = caseReports.map((report) => report.runs[report.runs.length - 1]);
  const axisMeans = Object.fromEntries(AXES.map((axis) => [
    axis,
    Number(avg(finalRuns.map((run) => run.scores[axis])).toFixed(3))
  ])) as Record<Axis, number>;

  const confidenceValues = finalRuns.flatMap((run) => AXES.map((axis) => run.confidences[axis]));
  const confidenceDistribution = {
    min: Number(Math.min(...confidenceValues).toFixed(3)),
    max: Number(Math.max(...confidenceValues).toFixed(3)),
    mean: Number(avg(confidenceValues).toFixed(3))
  };

  return { axisMeans, confidenceDistribution };
}

function axisBiasWarnings(suite: GoldenSuite, axisMeans: Record<Axis, number>) {
  const warnings: string[] = [];
  const values = AXES.map((axis) => axisMeans[axis]);
  const overallMean = avg(values);
  const warnMeanAbove = suite.axisBias?.warnMeanAbove ?? 2.4;
  const warnMeanDeltaAbove = suite.axisBias?.warnMeanDeltaAbove ?? 1.2;

  for (const axis of suite.axisBias?.watchAxes ?? AXES) {
    if (axisMeans[axis] > warnMeanAbove) {
      warnings.push(`${axis} mean ${axisMeans[axis]} exceeds ${warnMeanAbove}`);
    }
    if (axisMeans[axis] - overallMean > warnMeanDeltaAbove) {
      warnings.push(`${axis} mean delta ${(axisMeans[axis] - overallMean).toFixed(3)} exceeds ${warnMeanDeltaAbove}`);
    }
  }

  return warnings;
}

async function main() {
  const suitePath = path.resolve(root, argValue("suite") ?? "tests/golden-evaluations.json");
  const missionsPath = path.resolve(root, "missions/all_missions.json");
  const suite = await readJson<GoldenSuite>(suitePath);
  const missions = await readJson<Mission[]>(missionsPath);
  const caseFilter = argValue("case");
  const runs = Number(argValue("runs") ?? suite.defaults?.runs ?? 3);
  const outPath = path.resolve(
    root,
    argValue("out") ?? `reports/eval-stability-${new Date().toISOString().replace(/[:.]/g, "-")}.json`
  );

  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY is required for eval stability runs.");
  }

  const cases = suite.cases.filter((testCase) => !caseFilter || testCase.id === caseFilter);
  if (!cases.length) throw new Error(`No golden cases matched${caseFilter ? `: ${caseFilter}` : "."}`);

  const caseReports: CaseReport[] = [];

  for (const testCase of cases) {
    const mission = missions.find((item) => item.mission_id === testCase.mission_id);
    if (!mission) throw new Error(`Mission not found for case ${testCase.id}: ${testCase.mission_id}`);

    const caseRuns: CaseRun[] = [];
    for (let index = 0; index < runs; index += 1) {
      const result = await evaluateAnswer({ mission, answer: testCase.answer });
      caseRuns.push({
        run: index + 1,
        scores: scoresFromEvaluation(result.evaluation),
        confidences: confidencesFromEvaluation(result.evaluation),
        evidenceCounts: evidenceCountsFromEvaluation(result.evaluation),
        flags: result.evaluation.flags,
        prompt_version: result.evaluation.prompt_version
      });
    }

    const checked = checkCase(testCase, caseRuns, suite);
    const report: CaseReport = {
      id: testCase.id,
      mission_id: testCase.mission_id,
      tags: testCase.tags ?? [],
      status: checked.failures.length ? "fail" : "pass",
      failures: checked.failures,
      drift: checked.drift,
      runs: caseRuns
    };
    caseReports.push(report);

    const label = report.status === "pass" ? "PASS" : "FAIL";
    console.log(`${label} ${report.id}`);
    for (const failure of report.failures) console.log(`  - ${failure}`);
  }

  const { axisMeans, confidenceDistribution } = summarizeAxis(caseReports);
  const failedCases = caseReports.filter((report) => report.status === "fail");
  const warnings = axisBiasWarnings(suite, axisMeans);
  const maxScoreDrift = Math.max(...caseReports.map((report) => report.drift.maxScoreRange));
  const maxConfidenceDrift = Math.max(...caseReports.map((report) => report.drift.maxConfidenceRange));

  const report = {
    suite_version: suite.version,
    created_at: new Date().toISOString(),
    metadata: {
      prompt_version: PROMPT_VERSION,
      rubric_version: RUBRIC_VERSION,
      model_version: process.env.OPENAI_EVAL_MODEL ?? "gpt-4.1-mini",
      runs,
      suite_path: path.relative(root, suitePath)
    },
    summary: {
      total: caseReports.length,
      passed: caseReports.length - failedCases.length,
      failed: failedCases.length,
      axisMeans,
      axisBiasWarnings: warnings,
      maxScoreDrift,
      maxConfidenceDrift,
      confidenceDistribution,
      failedCases: failedCases.map((item) => ({
        id: item.id,
        failures: item.failures
      }))
    },
    cases: caseReports
  };

  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  console.log("");
  console.log(`Summary: ${report.summary.passed} passed, ${report.summary.failed} failed`);
  console.log(`Axis means: ${AXES.map((axis) => `${axis} ${axisMeans[axis]}`).join(" / ")}`);
  console.log(`Confidence: min ${confidenceDistribution.min}, mean ${confidenceDistribution.mean}, max ${confidenceDistribution.max}`);
  console.log(`Max drift: score ${maxScoreDrift}, confidence ${maxConfidenceDrift}`);
  if (warnings.length) console.log(`Axis bias warnings: ${warnings.join("; ")}`);
  console.log(`Report: ${path.relative(root, outPath)}`);

  if (failedCases.length && !hasArg("no-fail")) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`[eval-stability] ${message}`);
  process.exitCode = 1;
});
