"""
Back-Translation Data Augmentation for cs.AI Class

Mejora esperada: +1.5-2.5% accuracy
Técnica: Traducción EN→ES→EN para generar paráfrasis
Target: Duplicar muestras de cs.AI (450 → 900)
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from tqdm import tqdm
import os

class BackTranslationAugmenter:
    """
    Data augmentation mediante back-translation

    Process:
    1. English text → Spanish translation
    2. Spanish translation → English back-translation
    3. Result: Paraphrased version of original
    """

    def __init__(self, device=None):
        """
        Initialize translation models

        Uses MarianMT models from Helsinki-NLP
        - opus-mt-en-es: English to Spanish
        - opus-mt-es-en: Spanish to English
        """
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                      "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Loading translation models on {self.device}...")

        # English → Spanish
        print("  Loading EN→ES model...")
        self.model_en_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es')
        self.tokenizer_en_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es')
        self.model_en_es.to(self.device)
        self.model_en_es.eval()

        # Spanish → English
        print("  Loading ES→EN model...")
        self.model_es_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en')
        self.tokenizer_es_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')
        self.model_es_en.to(self.device)
        self.model_es_en.eval()

        print("✓ Translation models loaded\n")

    def translate(self, text, model, tokenizer, max_length=512):
        """
        Translate text using MarianMT model

        Args:
            text: Text to translate
            model: MarianMT model
            tokenizer: MarianMT tokenizer
            max_length: Maximum sequence length

        Returns:
            Translated text
        """
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Translate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_length,
                                    num_beams=4, early_stopping=True)

        # Decode
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

    def back_translate(self, text, max_length=512):
        """
        Perform back-translation: EN → ES → EN

        Args:
            text: Original English text
            max_length: Maximum sequence length

        Returns:
            Back-translated (paraphrased) English text
        """
        try:
            # EN → ES
            spanish = self.translate(text, self.model_en_es, self.tokenizer_en_es, max_length)

            # ES → EN
            back_translated = self.translate(spanish, self.model_es_en, self.tokenizer_es_en, max_length)

            return back_translated

        except Exception as e:
            print(f"Warning: Back-translation failed for text: {text[:50]}...")
            print(f"  Error: {e}")
            return text  # Return original if translation fails

    def augment_abstract(self, abstract):
        """
        Augment abstract via back-translation

        Strategy: Only augment abstract (not title) since:
        - Abstracts are longer (more content to paraphrase)
        - Titles are short and might lose meaning
        """
        return self.back_translate(abstract, max_length=512)

    def augment_dataset(self, df, target_category='cs.AI', augment_factor=1, max_samples_to_augment=450):
        """
        Augment dataset by duplicating and back-translating target category

        Args:
            df: DataFrame with columns [title, abstract, category]
            target_category: Category to augment (default: cs.AI)
            augment_factor: Number of augmented copies per sample (default: 1)
            max_samples_to_augment: Maximum number of samples to augment (default: 450)
                                   If None, augment all samples of target category

        Returns:
            Augmented DataFrame
        """
        print("="*70)
        print("DATASET AUGMENTATION")
        print("="*70)

        # Statistics
        total_before = len(df)
        target_count = len(df[df['category'] == target_category])

        print(f"\nOriginal dataset:")
        print(f"  Total samples: {total_before}")
        print(f"  {target_category} samples: {target_count}")
        print(f"\nAugmentation strategy:")
        print(f"  Target category: {target_category}")
        print(f"  Augment factor: {augment_factor}x")
        print(f"  Max samples to augment: {max_samples_to_augment if max_samples_to_augment else 'ALL'}")

        # Filter target category
        target_samples = df[df['category'] == target_category].copy()

        # Limit samples if specified
        if max_samples_to_augment and len(target_samples) > max_samples_to_augment:
            print(f"  ⚠️  Limiting to {max_samples_to_augment} random samples (out of {len(target_samples)})")
            target_samples = target_samples.sample(n=max_samples_to_augment, random_state=42)
            samples_to_augment = max_samples_to_augment
        else:
            samples_to_augment = len(target_samples)

        print(f"  Samples to augment: {samples_to_augment}")
        print(f"  Expected {target_category} after: {target_count + samples_to_augment * augment_factor}")
        print()

        # Create augmented copies
        augmented_samples = []

        for _ in range(augment_factor):
            print(f"Creating augmented copy {_+1}/{augment_factor}...")

            for idx, row in tqdm(target_samples.iterrows(), total=len(target_samples),
                                desc="  Back-translating"):
                augmented_abstract = self.augment_abstract(row['abstract'])

                augmented_samples.append({
                    'title': row['title'],  # Keep original title
                    'abstract': augmented_abstract,  # Augmented abstract
                    'category': row['category']
                })

        # Combine original + augmented
        augmented_df = pd.DataFrame(augmented_samples)
        final_df = pd.concat([df, augmented_df], ignore_index=True)

        # Shuffle
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Statistics
        total_after = len(final_df)
        target_after = len(final_df[final_df['category'] == target_category])

        print(f"\n✓ Augmentation complete!")
        print(f"\nFinal dataset:")
        print(f"  Total samples: {total_after} (+{total_after - total_before})")
        print(f"  {target_category} samples: {target_after} (+{target_after - target_count})")
        print(f"\nClass distribution:")
        print(final_df['category'].value_counts())
        print("="*70 + "\n")

        return final_df


def augment_arxiv_dataset(input_path='data/arxiv_papers_raw.csv',
                         output_path='data/arxiv_papers_augmented.csv',
                         target_category='cs.AI',
                         augment_factor=1,
                         max_samples_to_augment=450):
    """
    Main function to augment ArXiv dataset

    Args:
        input_path: Path to original CSV
        output_path: Path to save augmented CSV
        target_category: Category to augment
        augment_factor: Number of augmented copies per sample
        max_samples_to_augment: Maximum number of samples to augment (default: 450)
    """
    print("="*70)
    print("ARXIV DATASET AUGMENTATION PIPELINE")
    print("="*70)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Target: {target_category} (limit: {max_samples_to_augment} samples)")
    print(f"Augment factor: {augment_factor}x")
    print("="*70 + "\n")

    # Check input exists
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return None

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} samples\n")

    # Create augmenter
    augmenter = BackTranslationAugmenter()

    # Augment
    augmented_df = augmenter.augment_dataset(df, target_category, augment_factor, max_samples_to_augment)

    # Save
    print(f"Saving augmented dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(augmented_df)} samples\n")

    print("="*70)
    print("✓ AUGMENTATION COMPLETE")
    print("="*70)
    print(f"\nNext step: Train model on augmented data")
    print(f"  python train_scibert_v5_crossattn_aug.py")
    print("="*70 + "\n")

    return augmented_df


if __name__ == "__main__":
    # Example usage
    print(__doc__)

    # Run augmentation
    augmented_df = augment_arxiv_dataset(
        input_path='data/arxiv_papers_raw.csv',
        output_path='data/arxiv_papers_augmented.csv',
        target_category='cs.AI',
        augment_factor=1,  # Duplicate cs.AI samples
        max_samples_to_augment=450  # Limit to 450 samples for reasonable time (~50 min)
    )

    if augmented_df is not None:
        print("\n✓ Ready to train V5.0 with augmented data!")
