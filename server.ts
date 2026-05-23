import "dotenv/config";
import express from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { evaluateAnswer } from "./src/lib/evaluator/evaluateAnswer.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = Number(process.env.PORT ?? 8080);

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "app")));

app.post("/api/evaluate", async (req, res) => {
  try {
    const { mission, answer } = req.body ?? {};

    if (!mission || typeof answer !== "string") {
      return res.status(400).json({ error: "mission and answer are required." });
    }

    const result = await evaluateAnswer({ mission, answer });
    return res.json(result);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown evaluation error.";
    console.error("[evaluate]", message);
    return res.status(500).json({ error: message });
  }
});

app.listen(port, () => {
  console.log(`JOBSIM server running at http://localhost:${port}`);
});
