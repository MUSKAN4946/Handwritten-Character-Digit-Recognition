DIGIT_LABELS = [str(i) for i in range(10)]
LETTER_LABELS = [chr(ord("A") + i) for i in range(26)]
CLASS_NAMES = DIGIT_LABELS + LETTER_LABELS

def class_name(class_id: int) -> str:
    """Convert class index 0..35 into its displayed character."""
    if not 0 <= int(class_id) < len(CLASS_NAMES):
        raise ValueError("class_id must be between 0 and 35")
    return CLASS_NAMES[int(class_id)]
