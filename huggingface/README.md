---
title: ASD Detection
emoji: 🧩
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: AQ-10 ASD screening with a stacking ensemble
---

# ASD Detection System

Screens for Autism Spectrum Disorder (ASD) traits from the AQ-10 questionnaire,
using a stacking ensemble (SVM, k-NN and Decision Tree, with an XGBoost
meta-model). This is a screening aid, not a diagnosis.

Source code and technical report: see the project's GitHub repository.

Predictions are stored in Supabase when the `SUPABASE_URL` and `SUPABASE_KEY`
secrets are set in the Space settings.
