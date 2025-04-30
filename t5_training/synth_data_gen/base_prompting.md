### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a natural language instruction followed by a long biomedical text
- An `output`: a classification label ("Yes" or "No") followed by a brief but clear justification

### Output Format:
Each training example should be output as a JSON object on a single line (JSONL format), using the following structure:

```json
{
  "input": "<Instruction>\n\nText: <Brief biomedical paragraph>",
  "output": "<Yes/No>. <Justification based on content>"
}
```

---

### Instruction for This Batch:
**Instruction**: Does the paper use proteins as biomarkers and not genes or transcripts?

---

### Guidelines:
1. The paragraph (`Text`) should simulate realistic scientific writing, as found in biomedical abstracts or methods sections. The paragraph (`Text`) should be ~600 words long.
2. Roughly 50% of examples should be **Yes**, and 50% **No**.
3. Vary the phrasing of both the biomedical text and justification.
4. Include a wide variety of relevant biomarkers (e.g., "amyloid beta", "tau", "synaptophysin", "PSEN1", "MAPT", "IL-6").
5. Add "distractor" biological terms like genes, RNA, pathways, or irrelevant measurements (e.g., "astrocyte activation", "brain slices") to challenge the model.
6. The justification should reference *why* the answer is Yes or No with evidence from the text.

---

### Example:

```json
{
  "input": "Instruction: Does the paper use proteins as biomarkers and not genes or transcripts?\n\nText: Amyloid beta and total tau levels in cerebrospinal fluid were measured using ELISA in AD patients. No gene expression analysis was performed.",
  "output": "Yes. The study uses protein biomarkers (amyloid beta and tau) and does not mention gene or transcript-level analysis."
}
```

```json
{
  "input": "Instruction: Does the paper use proteins as biomarkers and not genes or transcripts?\n\nText: RNA sequencing was conducted to evaluate PSEN1 expression in hippocampal tissue. No protein levels were measured.",
  "output": "No. The paper focuses on gene expression and does not use proteins as biomarkers."
}
```
---