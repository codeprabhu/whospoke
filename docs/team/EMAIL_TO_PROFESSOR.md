# Draft email to the professor / TA

Send before the mid-semester presentation so that the changes read as agreed design decisions.

---

Subject: CS F407 project (Who Spoke What and When): two things to check before the mid-sem review

Dear Prof. Tirtharaj Dash,

We are Darshan Rajagoli, Vismay, Abhinav Padhi and Shrivaths Prabhu, working on the Advanced Computational
Speech Engineering project. We have finished Milestones 1 to 3 (separation, diarization and Hindi/Hinglish
transcription) and tested them. There are two changes from the proposal that we wanted to check with you
before the mid-sem review.

1. Order of Milestones 1 and 2. We built the order in the proposal (separate the whole recording first,
   then diarize), and we also tried doing it the other way round: diarize first, then separate only the
   parts where two people talk at the same time. We ran both on the same 72 test conversations. The second
   way worked better. The who-said-what word error rate (cpWER) went from 55.8% to 49.5%, and the
   diarization error rate went from 26.2% to 20.3%. The separator also does about twice as well when it only
   gets the overlapping parts (+9.6 dB SI-SDR instead of +4.5 dB). We plan to show both and use the second
   one as the main pipeline, but we can switch back to the original order if you prefer.

2. Datasets. We are mainly using IndicVoices and Project Vaani. Nirantar is only available as one ~200 GB
   archive with all 22 languages mixed together, so one of us ran it on Colab and kept just the Hindi
   (about 135 hours). When we looked at it, the Hindi part turned out to come from the same recordings as
   IndicVoices. 165 of the 514 speakers our test conversations use are in it, and some files are exactly
   the same. Building test conversations from it would let the same voices show up in testing twice, and
   since we aren't training any models, the extra data doesn't really help us. So we only used a sample of
   it (with our test speakers removed) as an extra check on the two ASR models. IndicConformer got 11.0% WER
   and IndicWav2Vec got 30.2%, which matches what we saw on Vaani. We couldn't find a public source for
   AIR-RS-DB. Real broadcasts also don't come with labels saying who spoke when, so we test on
   conversations we made ourselves from real IndicVoices speech and real background noise. That way we
   know exactly who said what and when.

We are planning to do the LLM post-processing part (Milestone 4) after the review. Please let us know if
you want us to change either of these.

Thank you,
Darshan Rajagoli (for the team)
Code and results: https://github.com/darshanrajagoli/whospoke
