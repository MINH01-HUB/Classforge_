# ClassForge: Graph Attention Network for Classroom Allocation

ClassForge is an intelligent classroom allocation system that uses Graph Attention Networks (GAT) to optimize student placement based on various factors including academic performance, social relationships, and psychological well-being.

## 🌟 Features

- **Graph Attention Network (GAT) Model**
  - Dual-head architecture for node and edge prediction
  - Configurable attention mechanisms
  - Real-time performance visualization
  - Multiple clustering algorithms support

- **Interactive Dashboard**
  - Real-time model parameter tuning
  - Performance metrics visualization
  - Network graph visualization
  - Attention weight analysis
  - Model comparison tools

- **Data Management**
  - CSV data import/export
  - Neo4j graph database integration
  - Data transformation pipeline
  - Student relationship simulation

- **Report Management**
  - Rich text report writing and editing
  - Upload and embed images in reports
  - Import and manage rubrics and assignment requirements
  - Export reports as PDF files

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- Neo4j Database
- Docker (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/MINH01-HUB/ClassForge.git
   cd ClassForge
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Database Setup**
   ```bash
   # Using Docker
   docker-compose up -d
   
   # Or install Neo4j Desktop from https://neo4j.com/download/
   ```

4. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

5. **Environment Configuration**
   Create a `.env` file in the backend directory:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   FLASK_APP=run.py
   FLASK_ENV=development
   ```

### Running the Application

1. **Start the Backend**
   ```bash
   cd backend
   flask run
   ```

2. **Start the Frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the Dashboard**
   Open `http://localhost:3000` in your browser

## 📊 Model Architecture

### GAT Model Components

- **Input Layer**: Student features (academic, social, psychological)
- **Attention Layers**: Multiple attention heads for relationship analysis
- **Output Layer**: 
  - Node predictions (student performance)
  - Edge predictions (relationship types)

### Configurable Parameters

- Number of attention heads
- Number of layers
- Hidden dimensions
- Dropout rate
- Clustering algorithm

## 📁 Project Structure

```
ClassForge/
├── backend/
│   ├── app/
│   │   ├── models/         # Database models
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   ├── utils/          # Utility functions
│   │   ├── templates/      # HTML templates
│   │   └── static/         # Static files
│   ├── requirements.txt
│   └── run.py
├── frontend/
│   ├── components/         # React components
│   ├── pages/             # Page components
│   └── public/            # Static assets
└── docker-compose.yml
```

## 📝 Data Format

### Input Data (CSV)
Required columns:
- student_id
- encoded_gender
- encoded_immigrant_status
- ses (Socioeconomic Status)
- achievement
- psychological_distress

### Output Data
- Node embeddings
- Edge predictions (Friend/Neutral/Conflict)
- Cluster assignments
- Performance metrics

## 🔧 API Endpoints

- `POST /api/gat/run`: Run the GAT model
- `PUT /api/gat/parameters`: Update model parameters
- `GET /api/gat/results`: Get model results
- `POST /api/data/upload`: Upload student data
- `GET /api/data/export`: Export results

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- PyTorch Geometric
- Neo4j
- React
- Flask

## 📞 Support

For support, email your.email@example.com or open an issue in the repository.

## 📝 Report Management Features

ClassForge now includes a comprehensive report management system designed for academic and project reporting:

- **Report Writing & Editing**: Use a rich text editor to compose and format your reports directly in the browser.
- **Image Upload & Embedding**: Upload images (e.g., diagrams, screenshots) and embed them within your report content.
- **Rubric & Assignment Requirements Management**: Import rubrics and assignment requirements (from DOCX, PDF, or text files) to reference and track criteria as you write.
- **PDF Export**: Export your completed report as a professionally formatted PDF file for submission or sharing.

### How to Use

1. Navigate to the `Report` section in the web app.
2. Use the editor to write and format your report.
3. Upload images using the image upload tool and insert them where needed.
4. Import rubrics/requirements to view them alongside your report.
5. When finished, click `Export as PDF` to download your report.
