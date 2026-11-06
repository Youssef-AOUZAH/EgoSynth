"""Referee prompts and score parsing shared by the pipeline and the standalone evaluator.

This module has no heavy dependencies (stdlib only), so it can be imported and
tested without a GPU.
"""

import re

REFEREE_SYSTEM_PROMPT = (
    "You are an elite Lead Robotics Vision Scientist and Chief Synthetic Data Inspector for a state-of-the-art "
    "Vision-Language-Action (VLA) laboratory. Your sole mandate is to ruthlessly evaluate and curate generated "
    "egocentric (first-person) video demonstrations intended to pretrain highly dexterous robotic manipulation models. "
    "Because neural networks are hypersensitive to flawed physical priors, any temporal hallucinations, kinematic anomalies, "
    "or geometric clipping in this training data will irreversibly corrupt the robot's physical understanding of the real world. "
    "You must act as an uncompromising, perfectionist quality-assurance referee."
)

REFEREE_USER_PROMPT = (
    "Critically evaluate this synthetic video sequence to determine its viability for our VLA training dataset. "
    "You must conduct a hyper-detailed frame-by-frame analysis based strictly on the following criteria:\n\n"
    "1. Kinematic Integrity & Hand Anatomy: Do the hands possess exactly five fingers with natural joint articulation? "
    "Penalize any structural morphing, alien geometry, or spontaneous generation of extra digits as the hand moves.\n"
    "2. Contact Physics & Grasp Mechanics: Analyze the exact moment of interaction between the hand and the object. "
    "There must be absolute zero geometric clipping (e.g., fingers phasing through solid handles or tables). "
    "The grip pre-shaping must be physically plausible for the object's apparent weight and shape.\n"
    "3. Temporal Consistency: Does the object maintain a stable volume and texture throughout the manipulation, "
    "or does it melt, warp, or spontaneously change structure?\n"
    "4. Spatial & Lighting Realism: Observe the contact shadows cast by the hand onto the object and the environment. "
    "Are they physically consistent?\n\n"
    "First, write a highly analytical, terse 3-to-4 sentence critique explicitly citing observations regarding anatomy, "
    "contact physics, and temporal stability. Be unforgiving in your assessment of generative artifacts.\n\n"
    "Finally, synthesize your observations into a definitive, granular 'Realism Score' from 0 to 100. "
    "Any video exhibiting blatant clipping or hallucinated anatomy must score below 40. "
    "Format the absolute end of your reply exactly as: 'Score: [XX]'."
)

# Matches "Score: 85", "Score: [85]", "score=72/100", "**Score: [90]**".
# A separator (":" or "=") is required so that prose such as "score of 35" is ignored.
_SCORE_RE = re.compile(r"score\s*[:=]\s*\[?\s*(\d{1,3})\b", re.IGNORECASE)


def parse_score(text):
    """Return the referee's integer score (0-100), or None if it cannot be read.

    The *last* match wins, because the prompt asks for the score at the very end
    of the reply. Unparseable or out-of-range scores return None: callers must
    not invent a default score.
    """
    matches = _SCORE_RE.findall(text or "")
    if not matches:
        return None
    value = int(matches[-1])
    return value if 0 <= value <= 100 else None
