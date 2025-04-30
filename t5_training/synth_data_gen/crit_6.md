---

### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a **fixed** natural language instruction (asking if the paper uses blood as an AD biomarker) followed by a paragraph of text (~300–600 words) resembling an abstract or methods summary.
- An `output`: either `"Yes"` or `"No"`, followed by a clear justification referencing specific details from the text.

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

> **If the term Blood occurs in the paper, does it use Blood as an AD biomarker?**

---

### Guidelines

1. **Positive Class ("Yes"/"relevant paper")**  
   - The text should show that blood (or its derivatives, such as serum or plasma) is used to measure AD biomarkers (e.g., beta-amyloid, tau).  
   - Indications may include phrases like "blood samples were analyzed," "serum was tested for AD biomarkers," or "plasma levels of amyloid were measured."

2. **Negative Class ("No"/"irrelevant paper")**  
   - Exclude examples where "blood" is mentioned solely in the context of circulatory assessments such as "blood pressure" or studies focusing on vascular or hypertension aspects.  
   - If the text includes keywords such as "blood pressure," "hypertension study," or "vascular health," these should lead to a classification of "No."

3. **Justification**  
   - The justification should clearly state the presence (or absence) of evidence that blood is used to sample AD biomarkers.  
   - For positive examples, the justification must indicate that blood-based biomarker sampling (e.g., serum or plasma assays) is central to the study.  
   - For negative examples, the justification should note that although blood is mentioned, it appears solely in contexts such as blood pressure or vascular assessments.

4. Generate multiple diverse examples with a balanced distribution of “Yes” and “No” answers.

---

### Example

```json
{
  "input": "Instruction: If the term Blood occurs in the paper, does it use Blood as an AD biomarker?\n\nText: In this study, blood samples were drawn from patients to determine the levels of beta-amyloid and phosphorylated tau. The research focused on establishing a correlation between these biomarkers and disease progression in Alzheimer's Disease. The analysis of serum provided significant insights into the pathological changes characteristic of AD.",
  "output": "Yes. The text clearly indicates that blood samples are used for biomarker analysis of beta-amyloid and tau, which are central to AD research."
}
```

```json
{
  "input": "Instruction: If the term Blood occurs in the paper, does it use Blood as an AD biomarker?\n\nText: The study examined the effects of high blood pressure on brain function among elderly participants. Blood pressure measurements were taken multiple times, and the focus was on assessing vascular health. Although blood was mentioned in the methods, there was no analysis of blood-based biomarkers related to Alzheimer's Disease.",
  "output": "No. Although blood is mentioned, its role is confined to blood pressure measurement and vascular assessment rather than biomarker sampling for AD."
}
```

Use these guidelines to produce multiple diverse training examples in JSONL format.