---

### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a **fixed** natural language instruction (asking whether the paper includes fluids from non-clinical models to perform its study) followed by a paragraph of text (~300–600 words) resembling an abstract or methods summary.
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

> **Does the paper include Fluids from Non-Clinical Models to perform its study?**

---

### Guidelines

1. **Positive Class ("Yes"/"relevant paper")**  
   - The text should clearly indicate that the study uses fluids (e.g., cerebrospinal fluid, blood, serum, plasma) obtained from non-clinical models (typically animal studies).  
   - Biomarkers measured in these fluids or references to their use (for example, in AD research) should confirm the relevance.

2. **Negative Class ("No"/"irrelevant paper")**  
   - If the text focuses on tissue samples—such as brain slices, biopsy samples, or histology—indicate that the study does not include fluids from non-clinical models.  
   - Keywords to look out for include "tissue," "histology," or "brain slice," which should lead to a "No" classification.

3. **Justification**  
   - The justification should clearly state why the study is classified as "Yes" (citing the use of fluids such as CSF, blood, serum, or plasma in animal models) or "No" (highlighting a focus on tissue samples).
  
4. Produce multiple diverse examples with a balanced distribution of “Yes” and “No” answers.

---

### Example

```json
{
  "input": "Instruction: Does the paper include Fluids from Non-Clinical Models to perform its study?\n\nText: In this experimental study, researchers collected cerebrospinal fluid (CSF) and blood samples from a well-established rodent model of Alzheimer’s Disease. The objective was to analyze the levels of beta-amyloid and tau proteins in these fluids using immunoassay techniques, clearly indicating a focus on fluid biomarkers. The study did not involve any tissue sample analysis, such as brain slice histology, to support its findings.",
  "output": "Yes. The text specifically describes the collection and analysis of CSF and blood from an animal model, without including tissue samples, thereby meeting the criteria."
}
```

```json
{
  "input": "Instruction: Does the paper include Fluids from Non-Clinical Models to perform its study?\n\nText: This study aimed to investigate the structural changes in brain tissue using detailed histological analyses. Researchers examined brain slices obtained from animal models and performed staining to highlight areas of neurodegeneration. Although there was a brief mention of serum in the introduction, the primary focus of the work was on tissue sample analysis through histology.",
  "output": "No. Despite a brief mention of serum, the core of the study is based on tissue sample analysis and histology, which does not meet the fluid-based criteria."
}
```

Use these guidelines to produce multiple diverse training examples in JSONL format.