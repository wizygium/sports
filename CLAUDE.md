# CLAUDE.md - AI Assistant Guide for Sports Repository

## Repository Overview

This is the **Roboflow Sports** repository, an open-source collection of computer vision tools designed for sports analytics. The project focuses on pushing the boundaries of object detection, image segmentation, keypoint detection, and foundational models in the context of sports video analysis.

**Repository URL**: https://github.com/roboflow/sports
**License**: MIT
**Python Version**: >=3.8
**Primary Framework**: Built on top of [Supervision](https://github.com/roboflow/supervision)

### Core Purpose
Provides reusable computer vision tools for:
- Ball detection and tracking
- Player detection and tracking
- Team classification (identifying which team players belong to)
- Jersey number recognition (OCR)
- Camera calibration and view transformation
- Pitch/field keypoint detection
- Advanced visualizations (radar views, Voronoi diagrams)

## Codebase Structure

```
sports/
├── README.md                          # Main repository documentation
├── LICENSE                            # MIT License
├── setup.py                           # Package installation configuration
├── CLAUDE.md                          # This file - AI assistant guide
├── .gitignore                         # Git ignore patterns
│
├── sports/                            # Main package directory
│   ├── __init__.py                    # Package initialization
│   │
│   ├── soccer/                        # ⚽ Soccer module (legacy structure)
│   │   └── config.py                  # SoccerPitchConfiguration
│   │
│   ├── handball/                      # 🤾 Handball module (feat/handball branch)
│   │   ├── config.py                  # CourtConfiguration (IHF standard)
│   │   ├── annotators.py              # Court drawing functions
│   │   ├── tools.py                   # GoalEventTracker
│   │   └── handball_court_geometry.py # D-shaped arc geometry
│   │
│   ├── basketball/                    # 🏀 Basketball module (feat/handball branch)
│   │   ├── config.py                  # CourtConfiguration (NBA/FIBA)
│   │   ├── annotators.py              # Court drawing functions
│   │   └── tools.py                   # Shot tracking utilities
│   │
│   ├── annotators/                    # Legacy annotators (to be deprecated)
│   │   ├── __init__.py
│   │   └── soccer.py                  # Soccer-specific annotators
│   │                                  # - draw_pitch()
│   │                                  # - draw_points_on_pitch()
│   │                                  # - draw_paths_on_pitch()
│   │                                  # - draw_pitch_voronoi_diagram()
│   │
│   ├── common/                        # Shared utilities across sports
│   │   ├── __init__.py
│   │   ├── core.py                    # MeasurementUnit enum (feat/handball)
│   │   ├── ball.py                    # BallTracker, BallAnnotator classes
│   │   ├── team.py                    # TeamClassifier using SigLIP + UMAP + KMeans
│   │   └── view.py                    # ViewTransformer for perspective transforms
│   │
│   └── configs/                       # Legacy configs (to be deprecated)
│       ├── __init__.py
│       └── soccer.py                  # SoccerPitchConfiguration dataclass
│                                      # - Pitch dimensions, vertices, edges
│
└── examples/                          # Example implementations
    └── soccer/                        # Soccer analysis example
        ├── README.md                  # Soccer-specific documentation
        ├── main.py                    # Main entry point with 6 modes
        ├── requirements.txt           # Additional dependencies (ultralytics, gdown)
        ├── setup.sh                   # Downloads models and sample videos
        ├── data/                      # Created by setup.sh (not in git)
        │   ├── *.pt                   # YOLO model weights
        │   └── *.mp4                  # Sample videos
        └── notebooks/                 # Jupyter notebooks for training
            ├── train_ball_detector.ipynb
            ├── train_player_detector.ipynb
            └── train_pitch_keypoint_detector.ipynb
```

## Key Components and Modules

### 1. Annotators (`sports/annotators/soccer.py`)

**Purpose**: Rendering soccer-specific visualizations

**Key Functions**:
- `draw_pitch()` - Renders a 2D soccer pitch with proper dimensions
- `draw_points_on_pitch()` - Draws player/object positions on the pitch
- `draw_paths_on_pitch()` - Draws movement trajectories
- `draw_pitch_voronoi_diagram()` - Visualizes team control areas

**Dependencies**: supervision, numpy, opencv-python

### 2. Ball Tracking (`sports/common/ball.py`)

**Classes**:

**`BallTracker`**:
- Uses a buffer of recent positions to filter false positives
- Selects detection closest to the centroid of recent positions
- Buffer size configurable (default: 10 frames)

**`BallAnnotator`**:
- Draws animated trail showing ball movement
- Uses color gradient (matplotlib 'jet' palette)
- Interpolates circle radius for visual effect

### 3. Team Classification (`sports/common/team.py`)

**`TeamClassifier`**:
- **Architecture**: SigLIP vision model → UMAP dimensionality reduction → KMeans clustering
- **Model**: `google/siglip-base-patch16-224` from HuggingFace
- **Process**:
  1. Extract features from player crops using SigLIP
  2. Reduce to 3D using UMAP
  3. Cluster into 2 teams using KMeans
- **Batching**: Configurable batch size (default: 32) for efficient processing
- **Usage Pattern**:
  1. Collect crops from multiple frames (stride sampling)
  2. Call `fit()` to train classifier
  3. Use `predict()` on individual frames

### 4. View Transformation (`sports/common/view.py`)

**`ViewTransformer`**:
- Computes homography matrix between source and target points
- `transform_points()` - Maps coordinates (e.g., player positions to pitch coordinates)
- `transform_image()` - Warps images using perspective transform
- **Use Case**: Convert camera view to top-down pitch view (radar)

### 5. Soccer Configuration (`sports/configs/soccer.py`)

**`SoccerPitchConfiguration`**:
- Dataclass with official pitch dimensions (in cm)
- 32 vertices defining pitch geometry
- Edge connections for rendering
- Color coding for different zones
- Labels for keypoints

**Key Dimensions**:
- Pitch: 120m × 70m (12000cm × 7000cm)
- Center circle radius: 9.15m
- Penalty box: 20.15m × 41m
- Goal box: 5.5m × 18.32m

## Development Workflow

### Installation

**Standard installation**:
```bash
pip install git+https://github.com/roboflow/sports.git
```

**Development installation** (editable):
```bash
git clone https://github.com/roboflow/sports.git
cd sports
pip install -e .
```

**For examples**:
```bash
cd examples/soccer
pip install -r requirements.txt
./setup.sh  # Downloads pre-trained models and sample videos
```

### Dependencies

**Core dependencies** (from `setup.py`):
- `supervision` - Core CV utilities (from Roboflow)
- `numpy` - Numerical computing
- `opencv-python` - Computer vision operations
- `transformers` - HuggingFace models (SigLIP)
- `umap-learn` - Dimensionality reduction
- `scikit-learn` - Clustering (KMeans)
- `tqdm` - Progress bars
- `sentencepiece` - Tokenization for transformers
- `protobuf` - Serialization

**Example-specific**:
- `ultralytics` - YOLO models
- `gdown` - Google Drive downloads

**Development**:
- `pytest` - Testing framework (in extras_require)

### Git Workflow

**Current Branch**: `claude/add-claude-documentation-L7oyF`

**Important**:
- This is a feature branch for adding Claude documentation
- Main branch tracking not specified in git config
- Follow standard GitHub Flow practices

**Active Development Branches**:
- `feat/handball` - Contains handball and basketball support (see below)

### In-Development Sports (feat/handball branch)

The `feat/handball` branch contains two additional sports implementations:

#### 🤾 Handball
- **Module**: `sports/handball/`
- **Configuration**: `CourtConfiguration` with IHF standard dimensions
  - Court: 40m × 20m (4000cm × 2000cm)
  - Goal area (6m line): D-shaped arc with 6m radius
  - Free throw line (9m line): D-shaped arc with 9m radius
  - Penalty mark: 7m line
- **Annotators**:
  - `draw_court()` - Renders handball court with D-shaped arcs
  - `draw_goals_on_court()` - Goals (circles), saves (squares), blocks (crosses)
  - `draw_points_on_court()` - Generic point visualization
  - `draw_paths_on_court()` - Player movement paths
- **Tools**: `GoalEventTracker` for tracking goal events

#### 🏀 Basketball
- **Module**: `sports/basketball/`
- **Configuration**: `CourtConfiguration` with NBA/FIBA standards
  - NBA: 94ft × 50ft (2865cm × 1524cm)
  - FIBA: 28m × 15m (2800cm × 1500cm)
  - Three-point arc, paint area, free throw line
- **Annotators**:
  - `draw_court()` - Renders basketball court with arcs
  - `draw_made_and_miss_on_court()` - Shot visualization
  - `draw_points_on_court()` - Generic point visualization
  - `draw_paths_on_court()` - Player movement paths
- **Tools**: Shot tracking and analysis utilities

**Note**: These features are in active development and not yet merged to main.

## Soccer Example Modes

The `examples/soccer/main.py` script supports 6 operational modes:

### 1. PITCH_DETECTION
- Detects soccer field boundaries and keypoints
- Uses YOLOv8 keypoint detection model
- Annotates 32 pitch keypoints

### 2. PLAYER_DETECTION
- Detects players, goalkeepers, referees, and ball
- 4 class IDs: BALL (0), GOALKEEPER (1), PLAYER (2), REFEREE (3)
- Uses bounding box annotations

### 3. BALL_DETECTION
- Specialized ball tracking with inference slicing
- Uses BallTracker to filter false positives
- Animated trail visualization
- Slice-based inference (640×640) for high-resolution videos

### 4. PLAYER_TRACKING
- Player detection + ByteTrack tracking
- Maintains consistent player IDs across frames
- Minimum 3 consecutive frames for track initialization

### 5. TEAM_CLASSIFICATION
- Combines detection, tracking, and team classification
- Two-phase process:
  1. **Training phase**: Collect crops with stride sampling (every 60 frames)
  2. **Inference phase**: Classify all frames
- Goalkeeper team assignment based on proximity to team centroids

### 6. RADAR
- Full pipeline: detection + tracking + team classification + view transform
- Renders top-down pitch view overlay
- Shows player positions on 2D pitch in real-time

## Key Conventions

### Code Style

1. **Type Hints**: Used throughout (typing, numpy.typing)
2. **Docstrings**: Google-style docstrings for all public functions/classes
3. **Error Handling**: Validates inputs with descriptive ValueError messages
4. **Imports**: Organized in standard order (stdlib, third-party, local)

### Naming Conventions

- **Classes**: PascalCase (e.g., `TeamClassifier`, `BallTracker`)
- **Functions**: snake_case (e.g., `draw_pitch`, `get_crops`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `BALL_CLASS_ID`, `STRIDE`)
- **Private**: Not used extensively (library is small and focused)

### Design Patterns

1. **Stateful Trackers**: Use buffers/deques for temporal smoothing
   - `BallTracker.buffer` - Recent ball positions
   - `BallAnnotator.buffer` - Recent positions for trail

2. **Two-Phase ML**: Training phase + inference phase
   - `TeamClassifier.fit()` then `TeamClassifier.predict()`

3. **Supervision Integration**: Heavy use of `sv.Detections`, `sv.KeyPoints`
   - All detections use Supervision's data structures
   - Consistent with Roboflow ecosystem

4. **Generator Pattern**: Video processing uses generators
   - Memory efficient for long videos
   - `sv.get_video_frames_generator()`

### Important Constants

**Soccer Example** (`examples/soccer/main.py`):
- `STRIDE = 60` - Frame sampling for training (every 60th frame)
- Class IDs: `BALL=0, GOALKEEPER=1, PLAYER=2, REFEREE=3`
- Model paths use `PARENT_DIR` for relative paths

**Pitch Config**:
- All dimensions in centimeters
- 32 keypoint vertices
- 32 edge connections

## Common Tasks for AI Assistants

### Adding a New Sport

1. Create `sports/configs/{sport}.py` with field/court configuration
2. Create `sports/annotators/{sport}.py` with sport-specific drawing functions
3. Add example in `examples/{sport}/` following soccer structure
4. Update main README.md with new datasets

### Modifying Detection Models

**Model Locations**: `examples/soccer/data/*.pt`
- Downloaded by `setup.sh`
- Training notebooks in `examples/soccer/notebooks/`

**To swap models**:
1. Update model path constants in `main.py`
2. Ensure class IDs match expected values
3. Test with `PLAYER_DETECTION` mode first

### Adding New Annotators

**Pattern to follow** (`sports/annotators/soccer.py`):
```python
def draw_custom_annotation(
    config: SoccerPitchConfiguration,
    detections: sv.Detections,
    # ... other params
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Clear docstring explaining purpose.

    Args:
        config: Pitch configuration
        detections: Detection data
        padding: Padding in pixels
        scale: Scaling factor
        pitch: Optional existing pitch image

    Returns:
        Annotated pitch image
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    # Your annotation logic here

    return pitch
```

### Testing

**Current State**:
- No test suite present (pytest in extras_require but no tests/ directory)
- Manual testing via example modes

**Recommended Testing Approach**:
1. Test on sample videos from `setup.sh`
2. Verify each mode runs without errors
3. Visual inspection of outputs
4. For new features, add pytest tests in `tests/` directory

### Working with Video Processing

**Performance Tips**:
1. Use `stride` parameter for training data collection
2. Consider `imgsz` parameter for YOLO (larger = more accurate but slower)
3. Use inference slicing for small objects (ball detection)
4. Batch processing for feature extraction (TeamClassifier)

**Memory Management**:
- Generators prevent loading entire video into memory
- Process frame-by-frame
- Use `sv.VideoSink` for writing output videos

## Important Notes for AI Assistants

### Dependencies on Roboflow Ecosystem

This repository is tightly integrated with Roboflow's ecosystem:
- **Supervision**: Core dependency for all CV operations
- **Roboflow Universe**: Dataset hosting
- **Inference**: Detection API (not used in current code but related)

When suggesting changes, prefer supervision utilities over raw OpenCV when possible.

### Model Licensing

**IMPORTANT**: The example uses YOLOv8 from Ultralytics:
- YOLOv8 is licensed under **AGPL-3.0**
- The sports analytics code is **MIT licensed**
- Combined demo has dual licensing (see examples/soccer/README.md)

**For Production Use**: Consider this licensing when deploying.

### Data Download Requirements

- Models and videos are NOT in git (large files)
- Must run `setup.sh` before using examples
- Uses Google Drive links (requires `gdown`)
- If links break, check Roboflow Universe for updated datasets

### Performance Considerations

**Device Selection**:
- `--device cpu` - Slowest, works everywhere
- `--device cuda` - NVIDIA GPUs (Linux/Windows)
- `--device mps` - Apple Silicon (M1/M2)

**Typical Processing Speed**:
- Real-time not achievable on CPU for high-res videos
- GPU recommended for RADAR mode (runs 3 models)

### Known Limitations

From README challenges section:
1. **Ball tracking**: Difficult due to small size, rapid movement
2. **Jersey numbers**: OCR hampered by blur, occlusion
3. **Player tracking**: Occlusions cause ID switches
4. **Re-identification**: Players leaving/re-entering frame
5. **Camera calibration**: Dynamic cameras, varying angles

### Roadmap Items

From `examples/soccer/README.md`:
- [ ] Add smoothing to eliminate flickering in RADAR mode
- [ ] Add notebook for offline data analysis

## File Modification Guidelines

### When Editing Core Library (`sports/`)

1. **Maintain backward compatibility** - This is a public library
2. **Add type hints** - Follow existing pattern
3. **Write docstrings** - Google-style format
4. **Test with examples** - Ensure soccer example still works
5. **Consider generalization** - Will this work for basketball, etc.?

### When Editing Examples (`examples/`)

1. **Keep modes separate** - Each mode should be self-contained
2. **Use constants** - Don't hardcode paths or IDs
3. **Follow generator pattern** - Yield frames for memory efficiency
4. **Document mode in README** - Add usage example and demo video

### When Adding Dependencies

1. **Update `setup.py`** - Add to install_requires
2. **Check licenses** - Ensure compatible with MIT
3. **Consider size** - Keep installation lightweight
4. **Pin versions cautiously** - Allow flexibility for users

## Useful Commands

### Development
```bash
# Install in editable mode
pip install -e .

# Install with test dependencies
pip install -e ".[tests]"

# Run example (after setup.sh)
cd examples/soccer
python main.py --source_video_path data/2e57b9_0.mp4 \
               --target_video_path output.mp4 \
               --device cuda \
               --mode RADAR
```

### Git Operations
```bash
# Check current status
git status

# Create feature branch
git checkout -b feature/your-feature

# Commit changes
git add .
git commit -m "Description of changes"

# Push to remote
git push -u origin feature/your-feature
```

## Datasets Reference

Available on Roboflow Universe:
- ⚽ Soccer player detection: football-players-detection-3zvbc
- ⚽ Soccer ball detection: football-ball-detection-rejhg
- ⚽ Soccer pitch keypoint detection: football-field-detection-f07vi
- 🏀 Basketball court keypoint detection: basketball-court-detection-2
- 🏀 Basketball jersey numbers OCR: basketball-jersey-numbers-ocr

**Data Source**: DFL - Bundesliga Data Shootout (Kaggle competition)

## Quick Reference: Class Hierarchy

```
TeamClassifier
  ├── features_model: SiglipVisionModel
  ├── processor: AutoProcessor
  ├── reducer: umap.UMAP
  └── cluster_model: KMeans

BallTracker
  └── buffer: deque (recent positions)

BallAnnotator
  ├── buffer: deque (for trail)
  ├── color_palette: sv.ColorPalette
  └── radius, thickness

ViewTransformer
  └── m: homography matrix (cv2.findHomography)

SoccerPitchConfiguration
  ├── dimensions: width, length, etc.
  ├── vertices: List[Tuple[int, int]] (32 points)
  ├── edges: List[Tuple[int, int]] (connections)
  └── labels, colors
```

## Contact and Contribution

- **Issues**: https://github.com/roboflow/sports/issues
- **Contributions**: Welcome! See README.md challenges section
- **Author**: Piotr Skalski (piotr.skalski92@gmail.com)
- **Organization**: Roboflow

---

**Last Updated**: 2025-12-25
**Repository Version**: 0.1.0
**For**: AI assistants (Claude, GPT, etc.)
