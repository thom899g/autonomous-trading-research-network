"""
Central configuration module for the Autonomous Trading Research Network.
Handles environment variables, Firebase initialization, and global settings.
"""

import os
from dataclasses import dataclass
from typing import Optional
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class FirebaseConfig:
    """Firebase configuration with validation"""
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_x509_cert_url: str
    
    @classmethod
    def from_env(cls) -> Optional['FirebaseConfig']:
        """Initialize from environment variables with validation"""
        try:
            private_key = os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n')
            
            config = cls(
                project_id=os.getenv('FIREBASE_PROJECT_ID', ''),
                private_key_id=os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
                private_key=private_key,
                client_email=os.getenv('FIREBASE_CLIENT_EMAIL', ''),
                client_id=os.getenv('FIREBASE_CLIENT_ID', ''),
                auth_uri=os.getenv('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                token_uri=os.getenv('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                auth_provider_x509_cert_url=os.getenv('FIREBASE_AUTH_PROVIDER_CERT_URL', ''),
                client_x509_cert_url=os.getenv('FIREBASE_CLIENT_CERT_URL', '')
            )
            
            # Validate required fields
            required_fields = ['project_id', 'private_key', 'client_email']
            for field in required_fields:
                if not getattr(config, field):
                    logging.warning(f"Missing required Firebase config: {field}")
                    return None
                    
            return config
            
        except Exception as e:
            logging.error(f"Failed to load Firebase config: {str(e)}")
            return None

@dataclass
class TradingConfig:
    """Trading system configuration"""
    # Exchange configuration
    exchange_id: str = "binance"
    testnet: bool = True
    
    # Trading parameters
    initial_capital: float = 10000.0
    max_position_size: float = 0.1  # 10% of capital
    max_drawdown: float = 0.2  # 20% max drawdown
    
    # Strategy parameters
    genetic_population_size: int = 100
    generations_per_day: int = 24
    market_regime_count: int = 7
    
    # Risk management
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    max_open_positions: int = 5

class Config:
    """Main configuration singleton"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize all configuration components"""
        self.firebase = FirebaseConfig.from_env()
        self.trading = TradingConfig()
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration integrity"""
        if not self.firebase:
            logging.warning("Firebase configuration not found. Real-time features will be disabled.")
        
        if self.trading.max_position_size > 0.5:
            logging.warning("Max position size exceeds recommended limit of 50%")
            self.trading.max_position_size = 0.5
    
    def is_firebase_available(self) -> bool:
        """Check if Firebase is properly configured"""
        return self.firebase is not None

# Global configuration instance
config = Config()