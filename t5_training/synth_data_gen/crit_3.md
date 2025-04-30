---

### Task: Generate High-Quality Instruction-Following Training Samples

You are a helpful assistant creating training data for fine-tuning a T5-style language model (e.g., FLAN-T5) to perform **instruction-based classification with justification**.

Generate diverse examples where each example contains:

- An `input`: consisting of a **fixed** natural language instruction (asking whether, if the paper mentions statistical analysis, the sample size exceeds 50) followed by a paragraph of text (~300–600 words) resembling an abstract or methods summary.
- An `output`: either `"Yes"` or `"No"`, followed by a clear justification referencing details from the text, particularly focusing on the statistical analysis and the reported sample size.

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

> **If the paper mentions some kind of statistical analysis, does the sample size exceed 50 (i.e., n >= 50)?**

---

### Guidelines

1. **Positive Class ("Yes"/"relevant paper")**  
   - The text should indicate that statistical analysis is performed and that the human sample size is greater than or equal to 50 (e.g., "n = 60", "n = 75", etc.).  
   - This indicates robust data collection supporting statistical inferences.

2. **Negative Class ("No"/"irrelevant paper")**  
   - The text might include statistical analysis but the sample size is below 50 (e.g., "n = 30", "n = 45").  
   - Note: Although sample size can sometimes be considered in context with other strengths of the study, for this criterion the focus is to check the explicit numeric threshold. If the sample size does not meet the threshold, assign the negative class.

3. **Justification**  
   - Clearly state why the example was assigned "Yes" (e.g., citing a sample size such as "n = 75" and mentioning statistical tests) or "No" (e.g., stating that the sample size (e.g., "n = 30") is below the threshold).

4. Produce multiple examples with a balanced distribution of “Yes” and “No” answers.

---

### Example

```json
{
  "input": "Instruction: If the paper mentions some kind of statistical analysis, does the sample size exceed 50 (i.e., n >= 50)?\n\nText: In this study, researchers conducted a comprehensive analysis of cognitive performance among elderly participants. A detailed statistical analysis, including multivariate regression and ANOVA tests, was performed on the collected data. The methods section specifies that the study recruited a total of 72 individuals from community centers across the city. Data collection included standardized cognitive assessments, and the statistical analysis was rigorously applied to validate the results.",
  "output": "Yes. The text clearly states that statistical analysis was performed and the sample size was 72, which exceeds the threshold of 50."
}
```

```json
{
  "input": "Instruction: If the paper mentions some kind of statistical analysis, does the sample size exceed 50 (i.e., n >= 50)?\n\nText: This pilot study aimed to explore preliminary trends in memory decline. Statistical comparisons were made using t-tests and chi-square tests to analyze the collected data. However, the study only involved 35 participants, all of whom were recruited from a single outpatient clinic. Despite the promising analytical approach, the reported sample size remained under the benchmark of 50.",
  "output": "No. Although statistical analysis was conducted, the study's sample size of 35 is below the threshold of 50."
}
```

Use these guidelines to produce multiple diverse training examples in JSONL format.