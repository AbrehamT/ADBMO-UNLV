---

### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a **fixed** natural language instruction (asking whether the paper is an original research article) followed by a paragraph of text that might resemble an abstract or a short summary (~300–600 words).  
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

> **Is the paper an original research article?**

---

### Guidelines

1. **Positive Class ("Yes"/"relevant paper")**  
   - Original research articles typically mention data collection or experimental design, methods, and statistical analyses.  
   - These signals indicate that the article is reporting new, original findings (rather than summarizing or commenting on others’ work).

2. **Negative Class ("No"/"irrelevant paper")**  
   - If the text highlights that it is a review, perspective, editorial, commentary, poster, preprint, or other non-original-research formats, it should be labeled "No".  
   - Mentions of “review article,” “systematic review,” “perspective,” “editorial,” or “opinion piece” are common indicators that the paper is **not** an original research article.  

3. **Justification**  
   - The justification should explicitly point to the presence or absence of relevant research features (data collection, methods, stats) or mention of non-research formats (e.g., review, perspectives, commentary) in the provided text.

4. Generate multiple examples with a balanced distribution of "Yes" and "No" answers.

---

### Example

```json
{
  "input": "Instruction: Is the paper an original research article?\n\nText: This study describes the recruitment of 250 participants from two hospitals, followed by a randomized clinical trial to test a novel treatment for chronic migraine. Details of statistical power calculations and methods for adverse event tracking are provided, indicating new data were collected and analyzed to address a specific hypothesis.",
  "output": "Yes. The text clearly states the collection of new patient data, a clinical trial design, and statistical analysis, all indicative of original research."
}
```

```json
{
  "input": "Instruction: Is the paper an original research article?\n\nText: This article presents a systematic review of recent advances in migraine therapies, compiling data from 52 studies published in the last five years. While the authors discuss results and synthesize findings across multiple trials, they do not present new, original data.",
  "output": "No. The paper is described as a systematic review, indicating it is summarizing existing findings rather than providing new experimental data."
}
```

Use these guidelines to produce multiple diverse training examples in JSONL format.