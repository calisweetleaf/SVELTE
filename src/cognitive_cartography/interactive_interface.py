# src/cognitive_cartography/interactive_interface.py
"""
Interactive Interface for SVELTE Framework.
Provides user-directed exploration, querying, and annotation capabilities
for tensor spaces and symbolic structures.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import asyncio

# Web framework dependencies
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ExplorationMode(Enum):
    """Modes of exploration interface."""
    GUIDED = "guided"
    FREE_FORM = "free_form"
    COMPARATIVE = "comparative"
    COLLABORATIVE = "collaborative"
    TUTORIAL = "tutorial"

class QueryType(Enum):
    """Types of queries supported."""
    PATTERN_SEARCH = "pattern_search"
    SIMILARITY = "similarity"
    ANOMALY = "anomaly"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"

class AnnotationType(Enum):
    """Types of annotations."""
    HYPOTHESIS = "hypothesis"
    OBSERVATION = "observation"
    QUESTION = "question"
    INSIGHT = "insight"
    BOOKMARK = "bookmark"
    WARNING = "warning"

@dataclass
class ExplorationState:
    """Current state of exploration session."""
    user_id: str
    session_id: str
    current_view: Dict[str, Any]
    navigation_history: List[Dict[str, Any]] = field(default_factory=list)
    bookmarks: List[Dict[str, Any]] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class QueryRequest:
    """Structure for query requests."""
    query_type: QueryType
    parameters: Dict[str, Any]
    target_data: str
    user_id: str
    session_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class Annotation:
    """Structure for user annotations."""
    id: str
    annotation_type: AnnotationType
    content: str
    position: Dict[str, Any]
    user_id: str
    session_id: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class NavigationController:
    """Controls navigation through tensor spaces and visualizations."""
    
    def __init__(self):
        """Initialize navigation controller."""
        self.navigation_stack = []
        self.current_position = None
        self.bookmarks = {}
        self.path_optimizer = PathOptimizer()
        
    def navigate_to(self, target: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Navigate to specific location in tensor space."""
        # Validate target
        if not self._validate_target(target):
            raise ValueError("Invalid navigation target")
        
        # Save current position to stack
        if self.current_position:
            self.navigation_stack.append({
                "position": self.current_position,
                "timestamp": datetime.now(timezone.utc),
                "session_id": session_id
            })
        
        # Update current position
        self.current_position = target
        
        # Generate navigation response
        response = {
            "success": True,
            "new_position": target,
            "can_go_back": len(self.navigation_stack) > 0,
            "suggested_actions": self._generate_suggestions(target)
        }
        
        logger.info(f"Navigation to {target.get('type', 'unknown')} completed for session {session_id}")
        return response
    
    def go_back(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Navigate back to previous position."""
        if not self.navigation_stack:
            return None
        
        previous = self.navigation_stack.pop()
        if previous["session_id"] == session_id:
            self.current_position = previous["position"]
            return previous["position"]
        
        # Session mismatch - restore stack
        self.navigation_stack.append(previous)
        return None
    
    def create_bookmark(self, name: str, description: str, session_id: str) -> str:
        """Create bookmark at current position."""
        if not self.current_position:
            raise ValueError("No current position to bookmark")
        
        bookmark_id = str(uuid.uuid4())
        self.bookmarks[bookmark_id] = {
            "id": bookmark_id,
            "name": name,
            "description": description,
            "position": self.current_position.copy(),
            "session_id": session_id,
            "created": datetime.now(timezone.utc)
        }
        
        return bookmark_id
    
    def navigate_to_bookmark(self, bookmark_id: str, session_id: str) -> Dict[str, Any]:
        """Navigate to bookmarked position."""
        if bookmark_id not in self.bookmarks:
            raise ValueError(f"Bookmark {bookmark_id} not found")
        
        bookmark = self.bookmarks[bookmark_id]
        return self.navigate_to(bookmark["position"], session_id)
    
    def _validate_target(self, target: Dict[str, Any]) -> bool:
        """Validate navigation target."""
        required_fields = ["type", "coordinates"]
        return all(field in target for field in required_fields)
    
    def _generate_suggestions(self, position: Dict[str, Any]) -> List[str]:
        """Generate navigation suggestions based on current position."""
        suggestions = []
        
        if position.get("type") == "tensor_space":
            suggestions.extend([
                "Explore entropy gradients",
                "View attention patterns",
                "Search for anomalies"
            ])
        elif position.get("type") == "symbolic_space":
            suggestions.extend([
                "Analyze pattern relationships",
                "Examine grammar structures",
                "Compare with other layers"
            ])
        
        return suggestions

class PathOptimizer:
    """Optimizes exploration paths for efficiency."""
    
    def __init__(self):
        """Initialize path optimizer."""
        self.visit_graph = {}
        self.efficiency_metrics = {}
    
    def record_visit(self, position: Dict[str, Any], duration: float):
        """Record visit to position for optimization."""
        pos_key = self._position_key(position)
        
        if pos_key not in self.visit_graph:
            self.visit_graph[pos_key] = {
                "visits": 0,
                "total_duration": 0,
                "connections": set()
            }
        
        self.visit_graph[pos_key]["visits"] += 1
        self.visit_graph[pos_key]["total_duration"] += duration
    
    def suggest_optimal_path(self, start: Dict[str, Any], 
                           end: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest optimal path between two positions."""
        # Simplified pathfinding - would implement A* or similar in production
        return [start, end]
    
    def _position_key(self, position: Dict[str, Any]) -> str:
        """Generate key for position."""
        return f"{position.get('type', '')}_{hash(str(position.get('coordinates', '')))}"

class QueryEngine:
    """Processes complex queries against tensor and symbolic data."""
    
    def __init__(self, data_sources: Dict[str, Any]):
        """Initialize query engine with data sources."""
        self.data_sources = data_sources
        self.query_cache = {}
        self.query_history = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def execute_query(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute query against available data sources."""
        # Check cache first
        cache_key = self._generate_cache_key(query)
        if cache_key in self.query_cache:
            logger.debug(f"Returning cached query result: {cache_key}")
            return self.query_cache[cache_key]
        
        # Execute query based on type
        start_time = time.time()
        
        if query.query_type == QueryType.PATTERN_SEARCH:
            result = self._execute_pattern_search(query)
        elif query.query_type == QueryType.SIMILARITY:
            result = self._execute_similarity_query(query)
        elif query.query_type == QueryType.ANOMALY:
            result = self._execute_anomaly_detection(query)
        elif query.query_type == QueryType.CAUSAL:
            result = self._execute_causal_analysis(query)
        elif query.query_type == QueryType.TEMPORAL:
            result = self._execute_temporal_analysis(query)
        elif query.query_type == QueryType.STRUCTURAL:
            result = self._execute_structural_analysis(query)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}")
        
        execution_time = time.time() - start_time
        
        # Add metadata to result
        result["query_metadata"] = {
            "execution_time": execution_time,
            "timestamp": datetime.now(timezone.utc),
            "cache_key": cache_key
        }
        
        # Cache result
        self.query_cache[cache_key] = result
        self.query_history.append(query)
        
        logger.info(f"Query executed in {execution_time:.2f}s: {query.query_type.value}")
        return result
    
    def _execute_pattern_search(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute pattern search query."""
        pattern = query.parameters.get("pattern", "")
        threshold = query.parameters.get("threshold", 0.8)
        
        matches = []
        target_data = self.data_sources.get(query.target_data, {})
        
        # Simplified pattern matching
        for key, data in target_data.items():
            if isinstance(data, np.ndarray):
                # Calculate pattern similarity (simplified)
                similarity = np.random.random()  # Placeholder for real pattern matching
                if similarity >= threshold:
                    matches.append({
                        "key": key,
                        "similarity": similarity,
                        "position": {"x": 0, "y": 0},  # Placeholder coordinates
                        "metadata": {"shape": data.shape}
                    })
        
        return {
            "matches": matches,
            "total_found": len(matches),
            "search_pattern": pattern
        }
    
    def _execute_similarity_query(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute similarity query."""
        reference = query.parameters.get("reference", "")
        similarity_metric = query.parameters.get("metric", "cosine")
        
        similarities = []
        target_data = self.data_sources.get(query.target_data, {})
        
        # Calculate similarities
        for key, data in target_data.items():
            if isinstance(data, np.ndarray) and key != reference:
                similarity = np.random.random()  # Placeholder for real similarity calculation
                similarities.append({
                    "key": key,
                    "similarity": similarity,
                    "metric": similarity_metric
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "similarities": similarities,
            "reference": reference,
            "metric": similarity_metric
        }
    
    def _execute_anomaly_detection(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute anomaly detection query."""
        threshold = query.parameters.get("threshold", 2.0)
        method = query.parameters.get("method", "statistical")
        
        anomalies = []
        target_data = self.data_sources.get(query.target_data, {})
        
        # Detect anomalies (simplified)
        for key, data in target_data.items():
            if isinstance(data, np.ndarray):
                # Statistical anomaly detection
                mean = np.mean(data)
                std = np.std(data)
                outliers = np.abs(data - mean) > threshold * std
                
                if np.any(outliers):
                    anomalies.append({
                        "key": key,
                        "anomaly_score": float(np.max(np.abs(data - mean) / std)),
                        "outlier_count": int(np.sum(outliers)),
                        "method": method
                    })
        
        return {
            "anomalies": anomalies,
            "threshold": threshold,
            "method": method
        }
    
    def _execute_causal_analysis(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute causal analysis query."""
        # Placeholder for causal analysis
        return {
            "causal_relationships": [],
            "confidence": 0.0,
            "method": "placeholder"
        }
    
    def _execute_temporal_analysis(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute temporal analysis query."""
        # Placeholder for temporal analysis
        return {
            "temporal_patterns": [],
            "time_range": query.parameters.get("time_range", "unknown"),
            "method": "placeholder"
        }
    
    def _execute_structural_analysis(self, query: QueryRequest) -> Dict[str, Any]:
        """Execute structural analysis query."""
        # Placeholder for structural analysis
        return {
            "structural_features": [],
            "topology": "unknown",
            "method": "placeholder"
        }
    
    def _generate_cache_key(self, query: QueryRequest) -> str:
        """Generate cache key for query."""
        query_str = f"{query.query_type.value}_{query.target_data}_{str(query.parameters)}"
        return str(hash(query_str))

class AnnotationSystem:
    """Manages user annotations and collaborative features."""
    
    def __init__(self):
        """Initialize annotation system."""
        self.annotations = {}
        self.annotation_index = {}
        self.lock = Lock()
        
    def create_annotation(self, annotation_type: AnnotationType, content: str,
                         position: Dict[str, Any], user_id: str, 
                         session_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new annotation."""
        annotation_id = str(uuid.uuid4())
        
        annotation = Annotation(
            id=annotation_id,
            annotation_type=annotation_type,
            content=content,
            position=position,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        with self.lock:
            self.annotations[annotation_id] = annotation
            self._update_index(annotation)
        
        logger.info(f"Created {annotation_type.value} annotation: {annotation_id}")
        return annotation_id
    
    def get_annotations_at_position(self, position: Dict[str, Any], 
                                   radius: float = 0.1) -> List[Annotation]:
        """Get annotations near a specific position."""
        annotations = []
        
        for annotation in self.annotations.values():
            if self._position_distance(annotation.position, position) <= radius:
                annotations.append(annotation)
        
        return annotations
    
    def get_annotations_by_user(self, user_id: str) -> List[Annotation]:
        """Get all annotations by a specific user."""
        return [ann for ann in self.annotations.values() if ann.user_id == user_id]
    
    def update_annotation(self, annotation_id: str, updates: Dict[str, Any], 
                         user_id: str) -> bool:
        """Update existing annotation."""
        if annotation_id not in self.annotations:
            return False
        
        annotation = self.annotations[annotation_id]
        
        # Check permissions
        if annotation.user_id != user_id:
            logger.warning(f"User {user_id} attempted to update annotation {annotation_id} owned by {annotation.user_id}")
            return False
        
        # Apply updates
        with self.lock:
            for key, value in updates.items():
                if hasattr(annotation, key):
                    setattr(annotation, key, value)
            
            self._update_index(annotation)
        
        return True
    
    def delete_annotation(self, annotation_id: str, user_id: str) -> bool:
        """Delete annotation."""
        if annotation_id not in self.annotations:
            return False
        
        annotation = self.annotations[annotation_id]
        
        # Check permissions
        if annotation.user_id != user_id:
            return False
        
        with self.lock:
            del self.annotations[annotation_id]
            self._remove_from_index(annotation)
        
        return True
    
    def search_annotations(self, query: str, annotation_type: Optional[AnnotationType] = None) -> List[Annotation]:
        """Search annotations by content."""
        results = []
        query_lower = query.lower()
        
        for annotation in self.annotations.values():
            if annotation_type and annotation.annotation_type != annotation_type:
                continue
            
            if query_lower in annotation.content.lower():
                results.append(annotation)
        
        return results
    
    def _position_distance(self, pos1: Dict[str, Any], pos2: Dict[str, Any]) -> float:
        """Calculate distance between two positions."""
        # Simplified Euclidean distance
        x1, y1 = pos1.get("x", 0), pos1.get("y", 0)
        x2, y2 = pos2.get("x", 0), pos2.get("y", 0)
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _update_index(self, annotation: Annotation):
        """Update annotation index."""
        # Simple indexing by position grid
        grid_x = int(annotation.position.get("x", 0) * 10)
        grid_y = int(annotation.position.get("y", 0) * 10)
        grid_key = f"{grid_x}_{grid_y}"
        
        if grid_key not in self.annotation_index:
            self.annotation_index[grid_key] = set()
        
        self.annotation_index[grid_key].add(annotation.id)
    
    def _remove_from_index(self, annotation: Annotation):
        """Remove annotation from index."""
        grid_x = int(annotation.position.get("x", 0) * 10)
        grid_y = int(annotation.position.get("y", 0) * 10)
        grid_key = f"{grid_x}_{grid_y}"
        
        if grid_key in self.annotation_index:
            self.annotation_index[grid_key].discard(annotation.id)

class CollaborationManager:
    """Manages multi-user collaboration features."""
    
    def __init__(self):
        """Initialize collaboration manager."""
        self.active_sessions = {}
        self.shared_sessions = {}
        self.user_permissions = {}
        self.session_lock = Lock()
        
    def create_shared_session(self, owner_id: str, session_name: str) -> str:
        """Create new shared collaboration session."""
        session_id = str(uuid.uuid4())
        
        with self.session_lock:
            self.shared_sessions[session_id] = {
                "id": session_id,
                "name": session_name,
                "owner": owner_id,
                "participants": {owner_id},
                "created": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "shared_state": {}
            }
            
            self.user_permissions[session_id] = {
                owner_id: {"read", "write", "admin"}
            }
        
        logger.info(f"Created shared session {session_id} owned by {owner_id}")
        return session_id
    
    def join_session(self, session_id: str, user_id: str) -> bool:
        """Join existing shared session."""
        if session_id not in self.shared_sessions:
            return False
        
        with self.session_lock:
            session = self.shared_sessions[session_id]
            session["participants"].add(user_id)
            session["last_activity"] = datetime.now(timezone.utc)
            
            # Grant default permissions
            if session_id not in self.user_permissions:
                self.user_permissions[session_id] = {}
            
            self.user_permissions[session_id][user_id] = {"read", "write"}
        
        logger.info(f"User {user_id} joined session {session_id}")
        return True
    
    def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave shared session."""
        if session_id not in self.shared_sessions:
            return False
        
        with self.session_lock:
            session = self.shared_sessions[session_id]
            session["participants"].discard(user_id)
            session["last_activity"] = datetime.now(timezone.utc)
            
            # Remove permissions
            if session_id in self.user_permissions:
                self.user_permissions[session_id].pop(user_id, None)
        
        return True
    
    def update_shared_state(self, session_id: str, user_id: str, 
                           state_update: Dict[str, Any]) -> bool:
        """Update shared session state."""
        if not self._check_permission(session_id, user_id, "write"):
            return False
        
        with self.session_lock:
            session = self.shared_sessions[session_id]
            session["shared_state"].update(state_update)
            session["last_activity"] = datetime.now(timezone.utc)
        
        return True
    
    def get_session_participants(self, session_id: str) -> List[str]:
        """Get list of session participants."""
        if session_id not in self.shared_sessions:
            return []
        
        return list(self.shared_sessions[session_id]["participants"])
    
    def _check_permission(self, session_id: str, user_id: str, permission: str) -> bool:
        """Check if user has specified permission."""
        if session_id not in self.user_permissions:
            return False
        
        user_perms = self.user_permissions[session_id].get(user_id, set())
        return permission in user_perms

class ExplorationSession:
    """Manages individual exploration sessions."""
    
    def __init__(self, user_id: str, data_sources: Dict[str, Any]):
        """Initialize exploration session."""
        self.session_id = str(uuid.uuid4())
        self.user_id = user_id
        self.data_sources = data_sources
        self.state = ExplorationState(user_id=user_id, session_id=self.session_id, current_view={})
        self.navigation = NavigationController()
        self.query_engine = QueryEngine(data_sources)
        self.annotations = AnnotationSystem()
        self.created = datetime.now(timezone.utc)
        self.last_activity = datetime.now(timezone.utc)
        
    def update_view(self, view_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update current view configuration."""
        self.state.current_view = view_config
        self.last_activity = datetime.now(timezone.utc)
        
        # Add to navigation history
        self.state.navigation_history.append({
            "view": view_config,
            "timestamp": self.last_activity
        })
        
        return {"success": True, "session_id": self.session_id}
    
    def execute_query(self, query_request: QueryRequest) -> Dict[str, Any]:
        """Execute query in this session."""
        query_request.user_id = self.user_id
        query_request.session_id = self.session_id
        
        result = self.query_engine.execute_query(query_request)
        self.last_activity = datetime.now(timezone.utc)
        
        return result
    
    def add_annotation(self, annotation_type: AnnotationType, content: str,
                      position: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add annotation to current session."""
        annotation_id = self.annotations.create_annotation(
            annotation_type, content, position, self.user_id, self.session_id, metadata
        )
        
        self.state.annotations.append(annotation_id)
        self.last_activity = datetime.now(timezone.utc)
        
        return annotation_id
    
    def create_bookmark(self, name: str, description: str) -> str:
        """Create bookmark at current position."""
        bookmark_id = self.navigation.create_bookmark(name, description, self.session_id)
        
        self.state.bookmarks.append({
            "id": bookmark_id,
            "name": name,
            "description": description,
            "created": datetime.now(timezone.utc)
        })
        
        return bookmark_id
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session activity."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created": self.created,
            "last_activity": self.last_activity,
            "navigation_history_length": len(self.state.navigation_history),
            "bookmarks_count": len(self.state.bookmarks),
            "annotations_count": len(self.state.annotations),
            "queries_executed": len(self.query_engine.query_history)
        }

class InteractiveInterface:
    """
    Main interactive interface for SVELTE framework.
    
    Provides comprehensive user interaction capabilities including
    exploration, querying, annotation, and collaboration features.
    """
    
    def __init__(self, data_sources: Dict[str, Any], port: int = 8080):
        """Initialize interactive interface."""
        self.data_sources = data_sources
        self.port = port
        self.active_sessions = {}
        self.collaboration_manager = CollaborationManager()
        self.connected_websockets = {}
        self.session_lock = Lock()
        
        # Initialize FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available. Web interface disabled.")
        
        logger.info(f"InteractiveInterface initialized on port {port}")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(title="SVELTE Interactive Interface", version="1.0.0")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app: FastAPI):
        """Add API routes to FastAPI app."""
        
        @app.get("/")
        async def root():
            return HTMLResponse(self._generate_html_interface())
        
        @app.post("/sessions")
        async def create_session(user_id: str):
            session = self.create_session(user_id)
            return {"session_id": session.session_id}
        
        @app.get("/sessions/{session_id}")
        async def get_session(session_id: str):
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            return self.active_sessions[session_id].get_session_summary()
        
        @app.post("/sessions/{session_id}/query")
        async def execute_query(session_id: str, query_data: dict):
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            query = QueryRequest(
                query_type=QueryType(query_data["query_type"]),
                parameters=query_data["parameters"],
                target_data=query_data["target_data"],
                user_id="",  # Will be set by session
                session_id=""  # Will be set by session
            )
            
            result = self.active_sessions[session_id].execute_query(query)
            return result
        
        @app.websocket("/ws/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await self._handle_websocket(websocket, session_id)
    
    def create_session(self, user_id: str) -> ExplorationSession:
        """Create new exploration session."""
        session = ExplorationSession(user_id, self.data_sources)
        
        with self.session_lock:
            self.active_sessions[session.session_id] = session
        
        logger.info(f"Created session {session.session_id} for user {user_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[ExplorationSession]:
        """Get existing session."""
        return self.active_sessions.get(session_id)
    
    def close_session(self, session_id: str) -> bool:
        """Close exploration session."""
        if session_id not in self.active_sessions:
            return False
        
        with self.session_lock:
            del self.active_sessions[session_id]
        
        logger.info(f"Closed session {session_id}")
        return True
    
    async def _handle_websocket(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connection."""
        await websocket.accept()
        
        if session_id not in self.active_sessions:
            await websocket.close(code=4004, reason="Session not found")
            return
        
        self.connected_websockets[session_id] = websocket
        
        try:
            while True:
                data = await websocket.receive_json()
                response = await self._process_websocket_message(data, session_id)
                await websocket.send_json(response)
        
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for session {session_id}")
        finally:
            self.connected_websockets.pop(session_id, None)
    
    async def _process_websocket_message(self, data: Dict[str, Any], 
                                       session_id: str) -> Dict[str, Any]:
        """Process incoming WebSocket message."""
        message_type = data.get("type", "unknown")
        session = self.active_sessions[session_id]
        
        if message_type == "update_view":
            return session.update_view(data.get("view_config", {}))
        
        elif message_type == "navigate":
            target = data.get("target", {})
            return session.navigation.navigate_to(target, session_id)
        
        elif message_type == "create_annotation":
            annotation_id = session.add_annotation(
                AnnotationType(data.get("annotation_type", "observation")),
                data.get("content", ""),
                data.get("position", {}),
                data.get("metadata", {})
            )
            return {"success": True, "annotation_id": annotation_id}
        
        elif message_type == "create_bookmark":
            bookmark_id = session.create_bookmark(
                data.get("name", ""),
                data.get("description", "")
            )
            return {"success": True, "bookmark_id": bookmark_id}
        
        else:
            return {"error": f"Unknown message type: {message_type}"}
    
    def _generate_html_interface(self) -> str:
        """Generate HTML interface."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVELTE Interactive Interface</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .panel { border: 1px solid #ccc; padding: 20px; margin: 10px 0; }
                .controls { display: flex; gap: 10px; margin: 10px 0; }
                button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
                button:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>SVELTE Interactive Interface</h1>
                
                <div class="panel">
                    <h2>Exploration Controls</h2>
                    <div class="controls">
                        <button onclick="createSession()">Create Session</button>
                        <button onclick="startExploration()">Start Exploration</button>
                        <button onclick="executeQuery()">Execute Query</button>
                    </div>
                </div>
                
                <div class="panel">
                    <h2>Visualization Area</h2>
                    <div id="visualization" style="height: 400px; border: 1px solid #eee;">
                        Visualization will appear here
                    </div>
                </div>
                
                <div class="panel">
                    <h2>Annotations</h2>
                    <div id="annotations">
                        <textarea placeholder="Add annotation..." rows="3" cols="50"></textarea>
                        <button onclick="addAnnotation()">Add Annotation</button>
                    </div>
                </div>
            </div>
            
            <script>
                let sessionId = null;
                let websocket = null;
                
                function createSession() {
                    fetch('/sessions', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({user_id: 'demo_user'})
                    })
                    .then(response => response.json())
                    .then(data => {
                        sessionId = data.session_id;
                        connectWebSocket();
                        alert('Session created: ' + sessionId);
                    });
                }
                
                function connectWebSocket() {
                    if (sessionId) {
                        websocket = new WebSocket(`ws://localhost:8080/ws/${sessionId}`);
                        websocket.onmessage = function(event) {
                            console.log('Received:', JSON.parse(event.data));
                        };
                    }
                }
                
                function startExploration() {
                    if (websocket) {
                        websocket.send(JSON.stringify({
                            type: 'update_view',
                            view_config: {type: 'tensor_space', layer: 0}
                        }));
                    }
                }
                
                function executeQuery() {
                    if (sessionId) {
                        fetch(`/sessions/${sessionId}/query`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                query_type: 'pattern_search',
                                parameters: {pattern: 'attention', threshold: 0.8},
                                target_data: 'tensor_field'
                            })
                        })
                        .then(response => response.json())
                        .then(data => console.log('Query result:', data));
                    }
                }
                
                function addAnnotation() {
                    if (websocket) {
                        const content = document.querySelector('textarea').value;
                        websocket.send(JSON.stringify({
                            type: 'create_annotation',
                            annotation_type: 'observation',
                            content: content,
                            position: {x: 0.5, y: 0.5}
                        }));
                    }
                }
            </script>
        </body>
        </html>
        """
    
    def start_server(self, host: str = "0.0.0.0"):
        """Start the interactive interface server."""
        if not self.app:
            raise RuntimeError("FastAPI not available. Cannot start web server.")
        
        logger.info(f"Starting SVELTE Interactive Interface on {host}:{self.port}")
        uvicorn.run(self.app, host=host, port=self.port)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())
    
    def cleanup_inactive_sessions(self, timeout_minutes: int = 60):
        """Clean up inactive sessions."""
        current_time = datetime.now(timezone.utc)
        timeout_delta = timedelta(minutes=timeout_minutes)
        
        to_remove = []
        for session_id, session in self.active_sessions.items():
            if current_time - session.last_activity > timeout_delta:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            self.close_session(session_id)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} inactive sessions")

def main():
    """CLI entry point for interactive interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SVELTE Interactive Interface")
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--demo', action='store_true', help='Start with demo data')
    args = parser.parse_args()
    
    # Create demo data sources
    data_sources = {
        "tensor_field": {
            "layer_0": np.random.rand(10, 10),
            "layer_1": np.random.rand(15, 15),
            "layer_2": np.random.rand(20, 20)
        },
        "entropy_maps": {
            "layer_0": np.random.rand(10, 10),
            "layer_1": np.random.rand(15, 15)
        }
    }
    
    interface = InteractiveInterface(data_sources, port=args.port)
    
    try:
        interface.start_server(host=args.host)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")

if __name__ == "__main__":
    main()