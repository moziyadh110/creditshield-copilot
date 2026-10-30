---
title: CreditShield Copilot
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Live backend for the CreditShield Copilot demo site
---

The live backend of [CreditShield Copilot](https://github.com/moziyadh110/creditshield-copilot).
Open the demo at https://moziyadh110.github.io/creditshield-copilot/ and press **Run live**.

This Space runs the real pipeline (input allow-list, Presidio, Reg B retrieval, the DeepSeek
draft, seven code checks, review queue) for the 50 eval cases only, with a per-visitor rate
limit and a $0.10 daily LLM budget. Simplified, not legal advice.
