---
layout: post
title: Data and Sampling 
date: 2024-01-08
description: Data and Sampling from a book Practical Statistics for Data Scientists 
tags: stattistics data-science
categories: data-science
giscus_comments: false
featured: true
related_posts: false
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/data_and_sampling.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/data_and_sampling.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
