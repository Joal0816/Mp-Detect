# core/location.py - GPS location service
"""GPS location service for microplastic detection sessions."""
from typing import Optional
from dataclasses import dataclass


@dataclass
class Location:
    """GPS location data."""
    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    altitude: Optional[float] = None
    timestamp: Optional[str] = None


class LocationService:
    """GPS location service for field testing."""
    
    def __init__(self):
        self._current_location: Optional[Location] = None
        self._permission_granted = False
    
    async def request_permission(self) -> bool:
        """Request location permission."""
        try:
            from flet import Geolocator, Permission
            geolocator = Geolocator()
            permission = await geolocator.request_permission()
            self._permission_granted = (
                permission == Permission.ALWAYS or 
                permission == Permission.WHILE_IN_USE
            )
            return self._permission_granted
        except ImportError:
            print("[Location] Geolocator not available")
            return False
        except Exception as e:
            print(f"[Location] Permission error: {e}")
            return False
    
    async def get_current_location(self) -> Optional[Location]:
        """Get current GPS location."""
        try:
            from flet import Geolocator
            geolocator = Geolocator()
            position = await geolocator.get_current_position()
            self._current_location = Location(
                latitude=position.latitude,
                longitude=position.longitude,
                accuracy=getattr(position, 'accuracy', None),
                altitude=getattr(position, 'altitude', None),
                timestamp=getattr(position, 'timestamp', None),
            )
            return self._current_location
        except ImportError:
            print("[Location] Geolocator not available")
            return None
        except Exception as e:
            print(f"[Location] Error: {e}")
            return None
    
    @property
    def latitude(self) -> Optional[float]:
        return self._current_location.latitude if self._current_location else None
    
    @property
    def longitude(self) -> Optional[float]:
        return self._current_location.longitude if self._current_location else None
    
    @property
    def is_available(self) -> bool:
        return self._current_location is not None
    
    def format_coordinates(self) -> str:
        """Format coordinates as string."""
        if not self._current_location:
            return "No location"
        lat = self._current_location.latitude
        lon = self._current_location.longitude
        lat_dir = "N" if lat >= 0 else "S"
        lon_dir = "E" if lon >= 0 else "W"
        return f"{abs(lat):.4f}° {lat_dir}, {abs(lon):.4f}° {lon_dir}"


# Singleton instance
location_service = LocationService()
