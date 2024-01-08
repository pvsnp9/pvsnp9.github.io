---
layout: post
title: Exploratory Data Analysis 
date: 2023-01-08
description: Exploring Basics of EDA from a book Practical Statistics for Data Scientists 
tags: stattistics data-science
categories: data-science
giscus_comments: false
related_posts: false
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/eda.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/eda.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}

