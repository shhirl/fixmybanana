# fixmybanana — structured output: prompt + schema (v1)

## 1. Response schema (JSON Schema, for OpenAI `response_format: { type: "json_schema", strict: true }`)

```json
{
  "name": "handstand_analysis",
  "strict": true,
  "schema": {
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "photo_usable": { "type": "boolean" },
      "unusable_reason": {
        "type": ["string", "null"],
        "enum": ["no_handstand", "not_side_view", "body_cut_off", "too_dark_or_blurry", "multiple_people", null]
      },
      "view": { "type": "string", "enum": ["side", "front", "back", "angled", "unknown"] },
      "support": { "type": "string", "enum": ["freestanding", "wall", "other", "unknown"] },
      "banana_score": { "type": "integer", "minimum": 0, "maximum": 10 },
      "banana_label": { "type": "string", "enum": ["straight_as_a_ruler", "slight_curve", "proper_banana", "full_croissant"] },
      "segments": {
        "type": "array",
        "items": {
          "type": "object",
          "additionalProperties": false,
          "properties": {
            "name": { "type": "string", "enum": ["hands_wrists", "shoulders", "ribs_upper_back", "hips_pelvis", "legs_feet"] },
            "status": { "type": "string", "enum": ["good", "minor", "major", "not_visible"] },
            "observation": { "type": "string" },
            "cue": { "type": ["string", "null"] }
          },
          "required": ["name", "status", "observation", "cue"]
        }
      },
      "primary_fix": { "type": "string" },
      "one_liner": { "type": "string" },
      "confidence": { "type": "string", "enum": ["low", "medium", "high"] },
      "confidence_reason": { "type": "string" }
    },
    "required": ["photo_usable", "unusable_reason", "view", "support", "banana_score", "banana_label",
                 "segments", "primary_fix", "one_liner", "confidence", "confidence_reason"]
  }
}
```

## 2. System prompt

```
You are a handstand form coach. You receive one photo of a person attempting a handstand and return a JSON analysis following the provided schema. Do not return anything outside the JSON.

Your job: rate how "banana" the handstand is (how far the body deviates from a straight vertical line from wrists to feet), identify which body segments cause it, and give one practical cue per problem segment.

Rules:
1. First decide if the photo is usable. Mark photo_usable=false and set unusable_reason if there is no handstand, the body is cut off, the image is too dark/blurry to judge alignment, or several people are in frame. If unusable, still fill every field: banana_score=0, banana_label="straight_as_a_ruler", segments all "not_visible", primary_fix and one_liner explain what photo to upload instead.
2. banana_score: 0 = perfectly stacked line, 10 = extreme arch/pike. Map to banana_label: 0–2 straight_as_a_ruler, 3–5 slight_curve, 6–8 proper_banana, 9–10 full_croissant.
3. Always return all five segments in order: hands_wrists, shoulders, ribs_upper_back, hips_pelvis, legs_feet. Use "not_visible" if you cannot see it. Cue is null when status is "good" or "not_visible".
4. Observations describe what you see ("ribs flare forward, lower back arched"). Cues are one short, actionable instruction ("pull ribs in and tuck pelvis, as if bracing for a punch"). Never give medical advice.
5. primary_fix names the single most important thing to work on. one_liner is a friendly, funny sentence for the top of the results card, banana-themed, max 20 words, never mocking body shape or ability.
6. Confidence: "high" only for a clear side view with full body visible. Front/back/angled views make arch hard to judge — use "low" or "medium" and say why in confidence_reason. Do not inflate confidence.
7. Mirrors and reflections count as the person. Wall-assisted handstands are fine; note support="wall" and judge alignment the same way.
8. If unsure between two scores, pick the lower banana_score and lower the confidence rather than guessing high.
```

## 3. Call shape (OpenAI, minimal)

```js
const res = await client.chat.completions.create({
  model: "gpt-4o-mini",          // or whatever you're on — cheap vision is fine here
  temperature: 0.2,
  messages: [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: [
      { type: "text", text: "Analyze this handstand." },
      { type: "image_url", image_url: { url: dataUrl, detail: "low" } }
    ]}
  ],
  response_format: { type: "json_schema", json_schema: HANDSTAND_SCHEMA }
});
const result = JSON.parse(res.choices[0].message.content);
```

## 4. What the UI does with it

- `photo_usable=false` → show `one_liner` + `primary_fix` as the "try another photo" message. Don't show a score.
- Results card: `one_liner`, big `banana_score`/`banana_label`, then `primary_fix`.
- Below: five segment rows, status as a colour dot, observation + cue. Hide cue when null.
- Small text: confidence + reason ("Medium — front view, arch is hard to judge"). This is the honesty signal.

## 5. Things to log per request (for the eval later)

photo hash, `view`, `support`, `banana_score`, `confidence`, model, latency, token cost, and any user feedback thumbs. That's enough to build the 30-photo eval from real traffic.

## 6. Design choices to explain on `/how-its-built` (from the drafting conversation)

- **`photo_usable` comes first.** The model must reject bad inputs explicitly instead of inventing a score for a photo of a cat. That is the guardrail.
- **Fixed five segments, always returned, with `not_visible` allowed.** A stable shape means the UI never breaks and the eval can compare like with like.
- **Confidence is forced to be honest.** Front views cannot get "high", and ties go to the lower score. Showing that on the results card is what makes the toy feel trustworthy.

Suggested first step after wiring it in: run it on 5–10 of Shirley's own handstand photos and note where it is wrong. That becomes iteration two of the prompt and the start of the eval.
