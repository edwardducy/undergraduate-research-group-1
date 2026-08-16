# Revisions

## Narrative Flow and Golden Thread Fixes

August 2026

This document records the fixes that bring title-defense-script.md in line with the golden thread defined in notes.md (Defense Section Mapping). The earlier content of this file recorded the unconditional method commitment. We applied that content, and it is no longer pending.

### The Golden Thread

Code-switched Taglish text fragments meaning, so the shared encoder must balance three tasks that pull it in different directions. No study has shown which balancing method prevents all three outputs from degrading. The study finds the consistently weakest output and designs a balancing method that prevents it from degrading.

### The Violations Found

1. The example beat played twice, near-verbatim. Part 1 (line 39) and Part 2 (line 67) both ended with the same three-clause beat: the location output loses, responders know that someone is stranded but not where, and the mistake delays or slows the rescue. The verbs even disagreed (delay versus slow).
2. The gap claim appeared three times. Line 55 ended with the claim, the bullet recap list restated it, and line 67 repeated it again.
3. The phrase "their results are also conditional" pointed at the Mazumder study, but the evidence that followed came from the Algerian study. The listener hears one study and gets evidence from another.
4. The phrase "the model" appeared in lines 39 and 67, which violates the terminology rule that bans "model" for our encoder.

### The Fixes Applied

1. Line 67 drops the example beat. The paragraph now ends with "Which one is an open question." Part 1 keeps the example, because there it is the mechanism illustration.
2. We removed the bullet recap list, together with its lead-in sentence. Each domain paragraph already ends with its own gap conclusion, so the list only repeated them. The consequence paragraph now follows the Philippine review directly.
3. "Their results are also conditional" becomes "Results in this area are also conditional."
4. "The model" becomes "the shared encoder" in line 39. The occurrence in line 67 disappeared with fix 1.

### Open Items

The golden thread's endpoint, the unconditional method commitment, lives only in notes.md. The script ends with the promise of the next section. Prepare the Q and A answer: the study always produces a balancing method that prevents the consistently weakest output from degrading.

### Pending Changes for Review

We have now applied both pending changes to the script. The record of what they fixed stays below for reference.

5. Line 39 contained an intra-paragraph redundancy. The sentence "The tasks must understand that "ma-evacuate" means evacuation, while NER must also track it exactly as written" restated the two sentences before it. Applied: the merge option. The example now lives inside the needs statements. "Intent classification and urgency prediction need the meaning of the tweet across both languages, so they must understand that "ma-evacuate" means evacuation. NER needs the words exactly as written, so it must track "ma-evacuate" with its exact spelling."

6. The Part 2 opening overlapped with the gap conclusions later in the section. The sentence "and no study applies dynamic multi-task balancing to joint triage of Taglish disaster tweets" repeated what the code-switched review and the consequence paragraph state. Applied: the trimmed opening. "Research already exists for each part of this study: disaster tweets, code-switched text, and Taglish. But no study combines these parts. We examine each part in turn."

Open item, no script change needed: the unconditional method commitment stays in notes.md until we write the methodology section of the full defense.
