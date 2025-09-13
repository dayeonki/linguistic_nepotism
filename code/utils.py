import re


def remove_citations(sentence):
    removed_sentence = re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sentence)).replace(" |", "").replace("]", "")
    return removed_sentence


def split_by_claims(sentence):
    claims, claims_wo_citations = [], []
    split_sentence = sentence.split(". ")
    for sent in split_sentence:
        if not sent.strip().endswith("."):
            sent += "."
        claims.append(sent)
        claims_wo_citations.append(remove_citations(sent))
    return claims, claims_wo_citations


def split_by_claims_bn(sentence):
    claims, claims_wo_citations = [], []
    split_sentence = sentence.split("। ")
    for sent in split_sentence:
        if not sent.strip().endswith("।"):
            sent += "।"
        claims.append(sent)
        claims_wo_citations.append(remove_citations(sent))
    return claims, claims_wo_citations


def split_by_claims_zh(sentence):
    claims, claims_wo_citations = [], []
    split_sentence = sentence.split("。")
    for sent in split_sentence:
        if not sent.strip().endswith("。"):
            sent += "。"
        claims.append(sent)
        claims_wo_citations.append(remove_citations(sent))
    return claims, claims_wo_citations


def make_prompt(
    question,
    context_title,
    context,
    prompt_template,
    language,
    total_words=200
):
    blocks = []
    for title, text, lang in zip(context_title, context, ['src'] * len(context)):
        blocks.append({"title": title, "text": text, "lang": lang})

    id_to_source = {}
    doc_blocks = []
    for i, block in enumerate(blocks, start=1):
        doc_blocks.append(f"Document ID: {i}\nTitle: {block['title']}\nContent: {block['text']}")
        id_to_source[str(i)] = block['lang']

    doc_section = "\n---\n".join(doc_blocks)
    prompt = prompt_template.replace("{{question}}", question)\
                            .replace("{{language}}", language)\
                            .replace("{{total_words}}", str(total_words))

    final_prompt = f"Information:\n{doc_section}\n---\n" + prompt
    return final_prompt, id_to_source


def make_prompt_main_run(
    context_title,
    context,
):
    blocks = []
    for title, text, lang in zip(context_title, context, ['en'] * len(context)):
        blocks.append({"title": title, "text": text, "lang": lang})

    id_to_source = {}
    doc_blocks = []
    for i, block in enumerate(blocks, start=1):
        doc_blocks.append(f"Document ID: {i}\nTitle: {block['title']}\nContent: {block['text']}")
        id_to_source[str(i)] = block['lang']

    doc_section = "\n---\n".join(doc_blocks)
    return doc_section


def make_prompt_main_run_miracl_n1(
    context_title,
    neg_context_title,
    context,
    neg_context,
):
    blocks = [{"title": context_title, "text": context, "lang": "en"}, {"title": neg_context_title, "text": neg_context, "lang": "src"}]

    doc_blocks = []
    for i, block in enumerate(blocks, start=1):
        doc_blocks.append(f"Document ID: {i}\nTitle: {block['title']}\nContent: {block['text']}")
    
    doc_section = "\n---\n".join(doc_blocks)
    return doc_section


def make_prompt_xrag(
    context_en,
    context_src,
    lang_1,
    lang_2
):
    blocks = [{"text": context_en, "lang": lang_1}, {"text": context_src, "lang": lang_2}]

    id_to_source = {}
    doc_blocks = []
    for i, block in enumerate(blocks, start=1):
        doc_blocks.append(f"Document ID: {i}\nContent: {block['text']}")
        id_to_source[str(i)] = block['lang']
    
    doc_section = "\n---\n".join(doc_blocks)
    return doc_section, id_to_source
