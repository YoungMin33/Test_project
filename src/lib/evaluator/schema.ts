import { z } from "zod";

export const AXES = ["AX1", "AX2", "AX3", "AX4", "AX5"] as const;
export const AxisSchema = z.enum(AXES);

export const EvidenceSchema = z.object({
  quote: z.string().min(1).describe("Short exact or near-exact evidence from the answer."),
  behavior: z.string().min(1).describe("Observed behavior represented by the quote."),
  level: z.number().int().min(1).max(4),
  primary_axis: AxisSchema,
  secondary_axes: z.array(AxisSchema).max(2),
  rationale: z.string().min(1).describe("Brief reason grounded in the quote.")
});

export const AxisResultSchema = z.object({
  score: z.number().int().min(0).max(4),
  confidence: z.number().min(0).max(1),
  evidence: z.array(EvidenceSchema),
  reason: z.string()
});

export const EvaluationSchema = z.object({
  mission_id: z.string(),
  axes: z.object({
    AX1: AxisResultSchema,
    AX2: AxisResultSchema,
    AX3: AxisResultSchema,
    AX4: AxisResultSchema,
    AX5: AxisResultSchema
  }),
  flags: z.array(z.enum([
    "too_short",
    "off_topic",
    "ambiguous",
    "low_confidence",
    "possible_prompt_injection"
  ])),
  prompt_version: z.string()
});

export type Axis = typeof AXES[number];
export type Evaluation = z.infer<typeof EvaluationSchema>;

export type MissionScore = Record<Axis, number>;
