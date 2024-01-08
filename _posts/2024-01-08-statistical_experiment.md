---
layout: post
title: Statistical Experiments and Significance Testing 
date: 2023-01-08
description: Statistical Experiments and Significance Testing from a book Practical Statistics for Data Scientists 
tags: stattistics data-science
categories: data-science
featured: true
giscus_comments: false
related_posts: false
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/statistical_experiment.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/statistical_experiment.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
