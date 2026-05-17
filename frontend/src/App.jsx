import { useState } from 'react';
import './App.css';

function App() {
  const [jdFolder, setJdFolder] = useState('');
  const [resumeFolder, setResumeFolder] = useState('');
  const [outputFolder, setOutputFolder] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState('');

  const browseFolder = async (setter) => {
    try {
      const res = await fetch('http://127.0.0.1:8000/api/select-folder');
      const data = await res.json();
      if (data.path) {
        setter(data.path);
      }
    } catch (err) {
      console.error(err);
      alert('Error connecting to backend to pick folder.');
    }
  };

  const handleMatch = async () => {
    if (!jdFolder || !resumeFolder || !outputFolder) {
      alert('Please select all folders first!');
      return;
    }
    setIsProcessing(true);
    setMessage('Matching resumes... This might take a few minutes.');
    try {
      const res = await fetch('http://127.0.0.1:8000/api/match', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jd_folder: jdFolder,
          resume_folder: resumeFolder,
          output_folder: outputFolder,
        }),
      });
      const data = await res.json();
      if (data.status === 'success') {
        setMessage('Match completed successfully! Check the output folder.');
      } else {
        setMessage('Error: ' + data.message);
      }
    } catch (err) {
      setMessage('Network error. Is the backend running?');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="container">
      <div className="card glass">
        <h1 className="title">AI Resume Matcher</h1>
        <p className="subtitle">Select folders to run the NLP matching algorithm</p>

        <div className="form-group">
          <label>Job Descriptions Folder</label>
          <div className="input-row">
            <input
              type="text"
              value={jdFolder}
              onChange={(e) => setJdFolder(e.target.value)}
              placeholder="Path to JDs..."
            />
            <button className="btn-secondary" onClick={() => browseFolder(setJdFolder)}>Browse</button>
          </div>
        </div>

        <div className="form-group">
          <label>Resumes Folder</label>
          <div className="input-row">
            <input
              type="text"
              value={resumeFolder}
              onChange={(e) => setResumeFolder(e.target.value)}
              placeholder="Path to Resumes..."
            />
            <button className="btn-secondary" onClick={() => browseFolder(setResumeFolder)}>Browse</button>
          </div>
        </div>

        <div className="form-group">
          <label>Output Folder</label>
          <div className="input-row">
            <input
              type="text"
              value={outputFolder}
              onChange={(e) => setOutputFolder(e.target.value)}
              placeholder="Path to save results..."
            />
            <button className="btn-secondary" onClick={() => browseFolder(setOutputFolder)}>Browse</button>
          </div>
        </div>

        <button
          className="btn-primary"
          onClick={handleMatch}
          disabled={isProcessing}
        >
          {isProcessing ? <span className="spinner"></span> : 'Run Matcher'}
        </button>

        {message && (
          <div className={`message-box ${message.includes('Error') || message.includes('error') ? 'error' : 'success'}`}>
            {message}
          </div>
        )}
      </div>

      <div className="background-decorations">
        <div className="blob-1"></div>
        <div className="blob-2"></div>
      </div>
    </div>
  );
}

export default App;
