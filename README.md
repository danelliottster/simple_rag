# simple_rag
A RAG implementation with base functionality meant to be self-hosted on a machine with modest resources.

# Introduction
Let's face it: RAGs are passe and 95% of use cases are incredibly basic.
The popular, publically-available, open-source RAG implementations which are available for self-hosting are rediculously overpowered.
Here we have a RAG implementation desgined for minimal functionality.

# Current features/tech stack

* Document parsing
  * Heavy reliance upon docling
* Document chunking
  * Heavily reliance upon chonkie
* Vector DB
  * SQLlite to hold the chunks and their emeddings
  * Scipy cKDTree for vector embedding search
* LLM interface
  * PydanticAI

# Roadmap

* Allow it to work with any LLM supplier which has an interface from PydanticAI
* Return citations
* Containerize

# Contribution
Contribution is welcome.  Shoot us a PR!

# How to use

## Setup environment

## Build a corpus of files

## Build DB

```
sqlite3 your_db_name.db < RAG/hugo_schema.sql

```

## Rebuild DB

## Test

### From the command line

### From the streamlit script

## Schedule a nightly DB rebuild

# Important entry points
