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

# Roadmap

# Contribution
Contribution is welcome.  Shoot us a PR!

# How to use

## Setup environment

