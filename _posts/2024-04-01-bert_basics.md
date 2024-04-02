---
layout: post
title: BERT Basics
date: 2024-04-01
description: Transformer for Natural Language Processing. Learn How to use pretrained BERT models and decipher their working.
tags: Transformer BERT NLP
categories: Transformer
featured: true
giscus_comments: false
related_posts: false
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/bert_basics.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/bert_basics.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
