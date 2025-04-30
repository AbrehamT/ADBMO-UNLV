---

### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a **fixed** natural language instruction (asking whether the paper looks at proteins as biomarkers, not genes/transcripts/fragments) followed by a paragraph of text (~300–600 words) resembling an abstract or methods summary.  
- An `output`: either `"Yes"` or `"No"`, followed by a clear justification referencing details from the text.

### Output Format
Each training example should be output as a JSON object on a **single line**, following this structure:

```json
{
  "input": "<Instruction>\n\nText: <Brief paragraph>",
  "output": "<Yes/No>. <Justification referencing the content>"
}
```

---

### **Instruction** (fixed for every example)

> **Does the paper look at proteins as biomarkers (not genes, nor transcripts nor fragments)?**

---

### Guidelines

1. **Positive Class ("Yes"/"relevant paper")**  
   - The text should include multiple references to proteins, such as "protein", "amyloid", "tau", or specific AD-related proteins (e.g., "beta-amyloid").  
   - The central context should be focused on measuring or analyzing proteins as biomarkers.

2. **Negative Class ("No"/"irrelevant paper")**  
   - If the text primarily focuses on non-protein entities like genes, RNA, transcription, or fragments, it should be classified as “No”.  
   - The presence of these terms multiple times and as central themes indicates a non-protein biomarker study.

3. **Justification**  
   - The justification should clearly state why the study is classified as "Yes" by referencing repeated mentions and focus on proteins.  
   - Conversely, it should highlight if the study mentions genes, transcripts, or fragments as the primary focus, leading to a "No" classification.

4. Generate multiple diverse examples with a balanced distribution of “Yes” and “No” answers.

---

### Example

```json
{
  "input": "Instruction: Does the paper look at proteins as biomarkers (not genes, nor transcripts nor fragments)?\n\nText: In this comprehensive study, researchers examined the levels of beta-amyloid and tau proteins in cerebrospinal fluid samples from patients diagnosed with Alzheimer’s Disease. Multiple protein assays were conducted to quantify these biomarkers, ensuring a detailed protein-based analysis. Although a brief mention of gene expression was noted, the bulk of the investigation centered on protein quantification through immunoassays and mass spectrometry techniques.",
  "output": "Yes. The paper predominantly focuses on protein biomarkers like beta-amyloid and tau, with only minimal and non-central reference to gene expression."
}
```

```json
{
  "input": "Instruction: Does the paper look at proteins as biomarkers (not genes, nor transcripts nor fragments)?\n\nText: This study explored the genetic basis of neurodegeneration by analyzing RNA samples extracted from brain tissues. A series of transcriptomic analyses were performed using next-generation sequencing. The research primarily focused on identifying gene expression profiles and transcriptional changes, with only passing references to protein levels which were not central to the analysis.",
  "output": "No. The central focus is on gene and transcript-level analysis, with proteins mentioned only tangentially, if at all."
}
```

Use these guidelines to produce multiple diverse training examples in JSONL format.