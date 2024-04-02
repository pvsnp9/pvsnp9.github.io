---
layout: post
title: QnA with Fine-Tuned BERT
date: 2024-04-01
description: Transformer for Natural Language Processing. Learn How to Fine-Tune BERT models for QnA and unserstand their working.
tags: QnA BERT NLP
categories: Transformer
featured: true
giscus_comments: false
related_posts: false
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/qa_fine_tune.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/qa_fine_tune.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
