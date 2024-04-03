---
layout: post
title: DistilBERT - Knowledge distillation from pretrained BERT
date: 2024-04-03
description:  How to distill knowledge from pretrained BERT model. A teacher student relationship.
tags: BERT DistilBERT
categories: Transformer
featured: true
giscus_comments: false
related_posts: false
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/knowledge_distillation.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/knowledge_distillation.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
