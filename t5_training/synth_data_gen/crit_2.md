---

### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a **fixed** natural language instruction (asking whether the paper’s main focus is Alzheimer’s Disease) followed by a paragraph of text (~300–600 words) resembling an abstract or summary.  
- An `output`: `"Yes"` or `"No"`, followed by a clear justification referencing details from the text.

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

> **Does the paper have Alzheimer’s Disease (AD) as its main focus?**

---

### Guidelines

1. **Positive Class ("Yes"/"relevant paper")**  
   - The text explicitly involves AD topics (diagnosis, treatment, pathology, etc.).  
   - The study includes or is about AD patients, including those with Mild Cognitive Impairment at risk of progressing to AD.  
   - The study references specific AD biomarkers (e.g., amyloid beta, tau).  

2. **Negative Class ("No"/"irrelevant paper")**  
   - The text does not mention Alzheimer’s Disease or focuses on other neurodegenerative conditions exclusively.  
   - Only general biomarkers or other neurodegeneration markers are mentioned (e.g., generic inflammatory markers) without tying them specifically to AD.  

3. **Justification**  
   - The justification should explicitly point to the presence (or absence) of AD-related details or biomarkers and clarify why that leads to a “Yes” or “No” classification.

4. Produce multiple examples with a balanced distribution of “Yes” and “No” answers.

---

### Example

```json
{
  "input": "Instruction: Does the paper have Alzheimer's Disease (AD) as its main focus?\n\nText: This study investigates the role of amyloid beta deposition in the hippocampus of patients diagnosed with early-stage Alzheimer’s Disease, assessing its correlation with cognitive decline using neuropsychological tests.",
  "output": "Yes. The text explicitly examines amyloid beta in patients with Alzheimer's Disease, confirming AD is the primary focus."
}
```

```json
{
  "input": "Instruction: Does the paper have Alzheimer's Disease (AD) as its main focus?\n\nText: Researchers performed an analysis of general tauopathies in older individuals and recorded multiple markers of neurodegeneration. While tau proteins are mentioned, the investigation primarily centers on age-related cognitive decline across diverse conditions, with no specific emphasis on AD.",
  "output": "No. Although tau is referenced, there is no clear focus on Alzheimer’s Disease; the study discusses a broader range of neurodegenerative conditions."
}
```

Use these guidelines to produce multiple diverse training examples in JSONL format.