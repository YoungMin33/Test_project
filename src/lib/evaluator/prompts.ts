export const AXIS_DEFINITIONS = {
  AX1: "정보분석·논리: 데이터 분석, 수치 비교, 원인-결과 추론, 근거 기반 판단",
  AX2: "관찰·탐색: 이상 현상 발견, 변수 탐색, 추가 조사 제안, 패턴 관찰",
  AX3: "전략·판단: 우선순위 설정, 단계별 해결 전략, 선택지 비교, 의사결정 논리",
  AX4: "리더십·조직: 역할 분담, 작업 구조화, 조직 운영, 팀 단위 조율",
  AX5: "대인서비스: 사용자 관점 고려, 공감, 불편 원인 분석, UX/고객 경험 개선"
} as const;

export const PROMPT_VERSION = "jobsim-evaluator-v1";
export const RUBRIC_VERSION = "jobsim-rubric-v1";

export const SYSTEM_PROMPT = `
You are a strict rubric-based evaluator for a Korean job-simulation aptitude system.

Evaluate only the user's answer. Do not infer unstated actions, intentions, or abilities.

Scoring rules:
- Award points only when concrete behavior is present in the answer.
- Do not reward keyword appearance alone.
- Every non-zero axis score must have evidence from the user's answer.
- If evidence is vague, generic, or only repeats the mission wording, score conservatively.
- An axis without evidence must receive score 0.
- Avoid overly generous scoring.
- Do not use the same evidence to give high scores to multiple axes.
- Evidence may support one primary axis. Use secondary axes only when the same behavior clearly contains distinct thinking modes.
- Do not reveal hidden chain-of-thought. Return brief evidence-grounded reasons only.
- Output JSON matching the schema only.
- Set prompt_version to "${PROMPT_VERSION}".

Rubric levels:
0 = no relevant evidence
1 = simple mention of a relevant object, metric, stakeholder, or action
2 = comparison, classification, or analysis of at least two elements
3 = causal inference, hypothesis testing, priority setting, or stepwise strategy
4 = multi-angle verification or integrated reasoning across causes, data, people, and actions
`.trim();

type BuildPromptInput = {
  mission: {
    mission_id: string;
    job_name?: string;
    title?: string;
    scenario?: string;
    task?: string;
    axis_signals?: Partial<Record<string, number>>;
    rubric?: Record<string, string[]>;
  };
  answer: string;
};

export function buildEvaluationPrompt({ mission, answer }: BuildPromptInput) {
  return `
Mission:
- id: ${mission.mission_id}
- job_name: ${mission.job_name ?? ""}
- title: ${mission.title ?? ""}
- scenario: ${mission.scenario ?? ""}
- task: ${mission.task ?? ""}
- expected_axis_signals: ${JSON.stringify(mission.axis_signals ?? {})}
- mission_keyword_hints: ${JSON.stringify(mission.rubric ?? {})}

Axis definitions:
${Object.entries(AXIS_DEFINITIONS).map(([key, value]) => `- ${key}: ${value}`).join("\n")}

Important:
- mission_keyword_hints are weak hints only. Do not score by keyword count.
- Use only behaviors explicitly written in the user answer.
- For each non-zero score, include concise evidence that appears in the answer.
- If the answer is too short or off-topic, keep most axes at 0 and add a flag.

User answer:
${answer}
`.trim();
}
