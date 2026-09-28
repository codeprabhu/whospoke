# Draft email to the professor / TA

Send before the mid-semester presentation so that the changes read as agreed design decisions.

---

**Subject:** CS F407 project (Who Spoke What and When) — two design decisions to confirm before the mid-sem review

Dear Prof. Tirtharaj Dash,

We are Darshan Rajagoli, Vismay, Abhinav Padhi and Shrivaths Prabhu, working on the Advanced Computational Speech Engineering project. Milestones 1–3
(separation, diarization, regional & code-switched transcription) are implemented and evaluated, and
we would like to flag two decisions before the mid-semester review:

1. **Order of Milestones 1 and 2.** We implemented the proposal's order (separate the whole recording,
   then diarize) and also an alternative (diarize first, then separate only the stretches where people
   overlap). On the same 72 test conversations the alternative lowers the who-said-what word error rate
   (cpWER) from 55.8 % to 49.5 % and the diarization error rate from 26.2 % to 20.3 %, and the separator itself
   works about twice as well when it only has to handle the overlaps (+9.6 dB vs +4.5 dB SI-SDR). We will present
   both and recommend the second, unless you would prefer we keep the original order as the main pipeline.

2. **Datasets.** We are using IndicVoices and Project Vaani. Nirantar comes as one ~200 GB archive with all
   the languages mixed together, so we can't download just the Hindi part. We also went through its
   metadata, and the Hindi part seems to come from the same collection as IndicVoices. It has the same
   fields, and 165 of the 514 speakers in the IndicVoices Hindi validation set (which our test conversations
   are built from) also appear in it. Some of the files are even identical. Still, one of us ran a Colab job
   that went through the whole archive and kept only the Hindi (about 135 hours). We tested both ASR models
   on a sample of it, after removing our test speakers. IndicConformer got 11.0% WER and IndicWav2Vec
   30.2%, the same ranking as on Vaani. Since it's the same collection as IndicVoices, we use it only as an
   extra check and not to build test conversations. We couldn't find a public
   source for AIR-RS-DB. Since real broadcasts don't come with labels, we test the pipeline on
   conversations we put together from real IndicVoices speech and real background noise, so we know
   exactly who said what and when.

The LLM post-processing stage (Milestone 4) is planned for after the review. Please let us know if you
would like either decision changed.

Thank you,
Darshan Rajagoli, on behalf of the team
(code and results: https://github.com/darshanrajagoli/whospoke)
