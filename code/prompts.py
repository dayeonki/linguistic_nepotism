get_report_prompt = """Using the above information, respond to the following query or task: "{{question}}".
The response should focus on the answer to the query, should be well structured, informative, and concise, with facts and numbers if available.

Please follow all of the following guidelines in your response:
- You MUST write in a single paragraph and at most {{total_words}} words.
- You MUST write the response in the following language: {{language}}.
- You MUST cite your sources, especially for relevant sentences that answer the question.
- When using information that comes from the documents, use citation which refer to the Document ID at the end of the sentence (e.g., [1]).
- Do NOT cite multiple documents at the end of the sentence (e.g., [1][2]).
- If multiple documents support the sentence, only cite the most relevant document.
- It is important to ensure that the Document ID is a valid string from the information above and that the information in the sentence is present in the document.

Response: """


get_claim_prompt = """Instruction: You are given a query, a document, and a sentence from a generated response that cites the document in answering the query. 
Determine which document best supports the information in the cited sentence. Respond only with the exact document ID. Do not provide any additional explanation

Query: "{{query}}"
Information:
"{{context}}"

Cited sentence: "{{claim}}"
Response: """


guess_citation_prompt = """Information: {{context}}
---
Using the above information, the response is the answer to the query or task: "{{query}}" in a single sentence.
You MUST cite the most relevant document by including only its Source ID in brackets at the end of the sentence (e.g., [Source ID]).
Do NOT include any additional words inside or outside the brackets.
Please output ONLY the number of the Source ID that is most relevant to the sentence.

Response: {{sentence}} ["""
