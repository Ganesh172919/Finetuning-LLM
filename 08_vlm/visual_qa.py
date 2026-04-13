"""
################################################################################
VISUAL QUESTION ANSWERING — ANSWERING QUESTIONS ABOUT IMAGES
################################################################################

########################################
CONCEPT EXPLANATION
########################################

What is Visual Question Answering?
    Answering natural language questions about images.

Architecture:
    Image + Question → Vision Encoder + Text Encoder → Answer

Interview Questions:
    Q: "How does VQA work?"
    A: Encode image and question, combine features, generate answer.

################################################################################
"""

import numpy as np

################################################################################
# SECTION 1: VQA MODEL
################################################################################

class VQAModel:
    """
    Visual Question Answering Model
    ================================

    Answers questions about images.

    Interview Questions:
        Q: "What are the challenges of VQA?"
        A: Compositional reasoning, counting, spatial relationships,
           common sense knowledge.
    """

    def __init__(self, d_vision: int = 256, d_text: int = 256, n_answers: int = 1000):
        self.d_vision = d_vision
        self.d_text = d_text

        # Vision encoder
        self.vision_proj = np.random.randn(d_vision, 256) * 0.02

        # Text encoder
        self.text_proj = np.random.randn(d_text, 256) * 0.02

        # Answer predictor
        self.answer_head = np.random.randn(256, n_answers) * 0.02

    def answer(
        self,
        image_features: np.ndarray,
        question_features: np.ndarray
    ) -> np.ndarray:
        """
        Answer question about image.

        Args:
            image_features: Vision features
            question_features: Text features

        Returns:
            logits: Answer logits
        """
        # Combine features
        img_emb = image_features @ self.vision_proj
        txt_emb = question_features @ self.text_proj
        combined = img_emb + txt_emb

        # Predict answer
        return combined @ self.answer_head


################################################################################
# SECTION 2: TESTING
################################################################################

def demonstrate_vqa():
    """Demonstrate VQA."""
    print("=" * 70)
    print("VISUAL QA DEMONSTRATION")
    print("=" * 70)

    model = VQAModel(d_vision=64, d_text=64, n_answers=100)
    img_feat = np.random.randn(64)
    q_feat = np.random.randn(64)
    logits = model.answer(img_feat, q_feat)
    print(f"Answer logits: {logits.shape}")

    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_vqa()
