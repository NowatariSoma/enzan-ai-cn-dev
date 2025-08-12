import logging
from typing import Optional
from common.helper import decode_token
from django.utils import timezone
from common.services import BaseService

from .models import Token, Invitation, User


class TokenService(BaseService):
    """Service class for handling token operations."""

    @staticmethod
    def validate_token(token: str, key: str = None) -> bool:
        """
        Validate if a token is valid and not expired.

        Args:
            token (str): The token to validate

        Returns:
            bool: True if token is valid, False otherwise
        """
        token_data = decode_token(token)
        if not token_data:
            return False

        return TokenService._is_token_valid_in_db(token, key)

    @staticmethod
    def validate_token_reset_password( token: str, key: str = None) -> bool:
        """
        Validate a token for resetting password.

        Args:
            token (str): The token to validate
            key (str): The key associated with the token

        Returns:
            bool: True if token is valid for resetting password, False otherwise
        """
        return TokenService._is_token_valid_in_db(token, key)

    @staticmethod
    def _is_token_valid_in_db(token: str, key: str = None) -> Optional[Token]:
        """
        Validate if a token exists in the database.

        Args:
            token (str): The token to validate

        Returns:
            Optional[Token]: The token object if found, None otherwise
        """
        db_token = Token.objects.filter(value=token).first()
        
        if not db_token or db_token.is_used or db_token.is_expired():
            return False
        
        if key and db_token.key != key:
            return False

        return True

    @staticmethod
    def mark_token_as_used(token_value: str) -> None:
        """
        Mark a token as used.

        Args:
            token_value (str): The token value to mark as used
        """
        Token.objects.filter(value=token_value).update(
            is_used=True,
            updated_at=timezone.now()
        )


class InvitationService(BaseService):
    """Service class for handling invitation operations."""

    @staticmethod
    def get_invitation(invitation_id: int) -> Optional[Invitation]:
        """
        Get an invitation by ID.

        Args:
            invitation_id (int): The invitation ID

        Returns:
            Optional[Invitation]: The invitation object if found, None otherwise
        """
        return Invitation.objects.filter(id=invitation_id).first()

    @staticmethod
    def validate_invitation(invitation: Optional[Invitation]) -> bool:
        """
        Validate if an invitation is valid and not expired.

        Args:
            invitation (Optional[Invitation]): The invitation to validate

        Returns:
            bool: True if invitation is valid, False otherwise
        """
        if not invitation:
            return False

        if invitation.registered or invitation.is_expired():
            return False

        if User.objects.filter(email=invitation.email).exists():
            return False

        return True


class UserService(BaseService):
    """Service class for handling user operations."""

    @staticmethod
    def get_user_by_email(email: str) -> Optional[User]:
        """
        Get a user by email.

        Args:
            email (str): The email of the user

        Returns:
            Optional[User]: The user object if found, None otherwise
        """
        return User.objects.filter(email=email).first()

    @staticmethod
    def get_user_by_id(user_id: int) -> Optional[User]:
        """
        Get a user by ID.

        Args:
            user_id (int): The ID of the user

        Returns:
            Optional[User]: The user object if found, None otherwise
        """
        return User.objects.filter(id=user_id).first()
