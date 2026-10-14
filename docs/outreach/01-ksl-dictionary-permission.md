# To the KSL Dictionary owners: permission for derived hand-pose data

**To:** [owner's name / contact from ksldictionary.com]
**Subject:** Request: using hand-movement data derived from KSL Dictionary videos in a free KSL learning app

---

Hello [name],

I'm Jerry Ochieng, a Kenyan software developer at Jarida in Nairobi. I'm building **Ashiria**, a free app for
Meta Quest headsets. It helps hearing families, teachers and classmates of Deaf people practise everyday Kenyan
Sign Language. Learners see 3D "shadow hands" demonstrating a sign and get feedback on each finger as they copy
it. It's a practice companion, made with paid Deaf co-creators. It is not a translator and does not replace
interpreters.

I'm entering it in Meta's VR Start Developer Competition (deadline 18 November 2026), and I plan to keep
developing it afterwards.

The KSL Dictionary is the best public reference for KSL I know of, and I'd like to ask your permission for one
specific use:

- **What we would use:** hand and arm joint positions (numbers, not pictures) that software measures from your
  videos of about 70 everyday signs: the alphabet, numbers 0–10, greetings, family and household words.
- **What we would not use:** no video, no images or stills, and none of your written descriptions would appear in
  the app, the competition video or any other material. The signer in your videos would not be shown or
  identifiable.
- **How it is used:** as first drafts of the 3D shadow hands and the "is this the right handshape?" checks. Our
  Deaf co-creators review every sign, and we replace drafts with our own recordings as we make them.
- **Where:** inside the Ashiria app on Meta Quest. During the competition that means a private test release of up
  to 200 people, including the judges. Later it may become a free public release.
- **Credit:** in the competition write-up and on the app's information page, in the words you prefer, for example
  "Sign references adapted, with permission, from the KSL Dictionary".
- **Your control:** you can withdraw permission at any time, and we will remove the derived data from the next
  version.

If you'd prefer something narrower, for example competition only, or only some signs, that works too. I'd also
be glad to share what we build with you, or to talk about working together more closely.

Could you reply to let me know whether this is acceptable, and under what conditions? A short email is enough. If
you'd like a formal agreement, I'm happy to sign one.

Thank you for the work you've put into the dictionary.

Kind regards,
Jerry Ochieng
Jarida · [phone] · [email]

---

**After a yes:** set `ksl_dictionary.permission: written`, `reference` and `date` in `content/rights.yaml`, and keep
the email. **After a no:** nothing changes in code. Studio recordings replace every draft (plan §5.4, option 2).
