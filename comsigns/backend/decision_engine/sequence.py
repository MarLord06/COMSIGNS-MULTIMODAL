"""
Sequence manager for tracking accepted words.

Maintains the current sequence state, tracking accepted
and rejected predictions in order of arrival.
"""

import logging
from typing import Dict, Any, Optional, List
from copy import deepcopy

from .types import (
    PredictionInput,
    AcceptanceResult,
    SequenceItem,
    RejectedItem,
    SequenceState
)

logger = logging.getLogger(__name__)


class SequenceManager:
    """Manages the sequence of accepted words.
    
    Tracks:
    - Accepted words in order of arrival
    - Rejected predictions with reasons
    
    Example:
        >>> manager = SequenceManager()
        >>> 
        >>> # Add an accepted prediction
        >>> acceptance = AcceptanceResult(accepted=True, ...)
        >>> prediction = PredictionInput(class_name="yo", ...)
        >>> manager.add_prediction(acceptance, prediction)
        >>> 
        >>> # Get current state
        >>> state = manager.get_state()
        >>> print(state.glosses)  # ["yo"]
    """
    
    def __init__(self):
        """Initialize empty sequence state."""
        self._state = SequenceState()
    
    def add_prediction(
        self,
        acceptance: AcceptanceResult,
        prediction: PredictionInput
    ) -> SequenceState:
        """Add a prediction to the sequence.
        
        If accepted, appends to the accepted list.
        If rejected, stores in rejected list with reason.
        
        Args:
            acceptance: Result of rule evaluation
            prediction: The original prediction input
        
        Returns:
            Current sequence state after addition
        """
        if acceptance.accepted:
            # Create sequence item
            item = SequenceItem(
                gloss=prediction.class_name,
                class_id=prediction.class_id,
                confidence=acceptance.confidence,
                bucket=acceptance.bucket,
                position=len(self._state.accepted)
            )
            self._state.accepted.append(item)
            
            logger.debug(
                f"Accepted: '{item.gloss}' at position {item.position} "
                f"({item.confidence:.2%})"
            )
        else:
            # Create rejected item
            rejected = RejectedItem(
                gloss=prediction.class_name,
                class_id=prediction.class_id,
                confidence=acceptance.confidence,
                bucket=acceptance.bucket,
                reason=acceptance.reason,
                rule_applied=acceptance.rule_applied
            )
            self._state.rejected.append(rejected)
            
            logger.debug(
                f"Rejected: '{rejected.gloss}' - {rejected.reason}"
            )
        
        return self._state
    
    def get_state(self) -> SequenceState:
        """Get the current sequence state.
        
        Returns:
            Copy of current state
        """
        return deepcopy(self._state)
    
    def get_accepted_glosses(self) -> List[str]:
        """Get list of accepted glosses in order.
        
        Returns:
            List of gloss strings
        """
        return [item.gloss for item in self._state.accepted]
    
    def get_sequence_length(self) -> int:
        """Get number of accepted words.
        
        Returns:
            Length of accepted sequence
        """
        return len(self._state.accepted)
    
    def get_last_accepted(self) -> Optional[SequenceItem]:
        """Get the last accepted word.
        
        Returns:
            Last SequenceItem or None if empty
        """
        if self._state.accepted:
            return self._state.accepted[-1]
        return None
    
    def reset(self) -> None:
        """Reset the sequence state.
        
        Clears all accepted and rejected items.
        """
        self._state = SequenceState()
        logger.debug("Sequence state reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary.
        
        Returns:
            Dictionary representation of state
        """
        return self._state.to_dict()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the sequence for API response.
        
        Returns:
            Simplified dictionary with key info
        """
        return {
            "accepted": [
                {
                    "gloss": item.gloss,
                    "confidence": round(item.confidence, 4),
                    "bucket": item.bucket
                }
                for item in self._state.accepted
            ],
            "length": len(self._state.accepted),
            "glosses": self.get_accepted_glosses()
        }
