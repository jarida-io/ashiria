# To Meta Developer Support / office hours: team-recorded reference hands

Ask through a Meta Developer Support ticket, the developer forum, or VR Start office hours. Keep the written answer.

**Subject:** Hand tracking data policy: may our team record our own hands to author reference poses shipped in the app?

---

Hello,

I'm building a Kenyan Sign Language practice app for Quest 3/3S (Unity, Meta XR SDK v207) for the VR Start
Developer Competition. I'd like to confirm that our content workflow fits the hand-tracking data policy, which says
hand data "is only permitted to be used for enabling hand tracking within your app".

**What we do:**
1. **Authoring, by the team only.** Members of our development team, including Deaf signers who are on the
   competition team, wear our headset running a development build. They sign each word several times, and the build
   saves the hand-joint poses to the headset. We copy them to our computer over USB and turn them into reference
   templates (joint positions over time), which ship inside the app. This is like authoring poses with the
   Interaction SDK's Hand Pose Recorder. The people recording give written consent.
2. **End users.** The app compares a user's live hand-tracking data with the shipped templates on the headset to
   give feedback such as "bend your ring finger". Nothing is uploaded, stored off the device, or shared. We keep
   only the user's own progress and their personal tolerance settings on the headset, and they can delete them.

**Questions:**
- Is (1), team members recording their own hands with a development build to author shipped reference poses,
  permitted?
- Is it fine for (2) to adapt a user's own tolerances on the device from their own attempts?
- Does anything need to be declared in the data use checkup or privacy policy beyond "hand tracking is processed
  on the device and never leaves it"?

Thank you,
Jerry Ochieng
[organisation / app id once created] · [email]

---

**Record the answer** in `docs/HACKATHON_PLAN.md` §5.4.
