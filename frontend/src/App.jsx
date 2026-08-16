import { useState } from 'react';
import './App.css';

function App() {
  const [jdFiles, setJdFiles] = useState([]);
  const [resumeFiles, setResumeFiles] = useState([]);

  const [isProcessing, setIsProcessing] = useState(false);
  const [message, setMessage] = useState('');

  // ==============================
  // HANDLE FOLDER SELECTION
  // ==============================

  const handleFolderSelect = (event, setter) => {
    const files = Array.from(event.target.files);

    setter(files);
  };

  // ==============================
  // RUN MATCHER
  // ==============================

  const handleMatch = async () => {

    if (jdFiles.length === 0 || resumeFiles.length === 0) {
      alert('Please select both JD and Resume folders first!');
      return;
    }

    setIsProcessing(true);

    setMessage(
      'Matching resumes... This might take a few minutes.'
    );

    try {

      const formData = new FormData();

      // Add JD files
      jdFiles.forEach((file) => {
        formData.append('jd_files', file);
      });

      // Add Resume files
      resumeFiles.forEach((file) => {
        formData.append('resume_files', file);
      });

      // Top K
      formData.append('top_k', '5');

      const response = await fetch(
        'http://127.0.0.1:8000/api/match',
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        throw new Error('Matching failed');
      }

      // ==============================
      // RECEIVE ZIP
      // ==============================

      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);

      const link = document.createElement('a');

      link.href = url;
      link.download = 'Matched_Resumes.zip';

      document.body.appendChild(link);

      link.click();

      link.remove();

      window.URL.revokeObjectURL(url);

      setMessage(
        'Match completed successfully! Matched_Resumes.zip has been downloaded.'
      );

    } catch (err) {

      console.error(err);

      setMessage(
        'Network error. Is the backend running?'
      );

    } finally {

      setIsProcessing(false);

    }
  };


  // ==============================
  // UI
  // ==============================

  return (
    <div className="container">

      <div className="card glass">

        <h1 className="title">
          AI Resume Matcher
        </h1>

        <p className="subtitle">
          Select folders to run the AI-powered matching algorithm
        </p>


        {/* ==============================
            JD FOLDER
        ============================== */}

        <div className="form-group">

          <label>
            Job Descriptions Folder
          </label>

          <div className="input-row">

            <input
              type="text"
              value={
                jdFiles.length > 0
                  ? `${jdFiles.length} files selected`
                  : ''
              }
              placeholder="Select JDs folder..."
              readOnly
            />

            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              id="jd-folder"
              onChange={(e) =>
                handleFolderSelect(
                  e,
                  setJdFiles
                )
              }
              style={{ display: 'none' }}
            />

            <label
              htmlFor="jd-folder"
              className="btn-secondary"
            >
              Browse
            </label>

          </div>

        </div>


        {/* ==============================
            RESUME FOLDER
        ============================== */}

        <div className="form-group">

          <label>
            Resumes Folder
          </label>

          <div className="input-row">

            <input
              type="text"
              value={
                resumeFiles.length > 0
                  ? `${resumeFiles.length} files selected`
                  : ''
              }
              placeholder="Select Resumes folder..."
              readOnly
            />

            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              id="resume-folder"
              onChange={(e) =>
                handleFolderSelect(
                  e,
                  setResumeFiles
                )
              }
              style={{ display: 'none' }}
            />

            <label
              htmlFor="resume-folder"
              className="btn-secondary"
            >
              Browse
            </label>

          </div>

        </div>


        {/* ==============================
            RUN BUTTON
        ============================== */}

        <button
          className="btn-primary"
          onClick={handleMatch}
          disabled={isProcessing}
        >

          {isProcessing
            ? <span className="spinner"></span>
            : 'Run Matcher'
          }

        </button>


        {/* ==============================
            MESSAGE
        ============================== */}

        {message && (

          <div
            className={`message-box ${
              message.includes('Error') ||
              message.includes('error') ||
              message.includes('Network')
                ? 'error'
                : 'success'
            }`}
          >
            {message}
          </div>

        )}

      </div>


      {/* ==============================
          BACKGROUND
      ============================== */}

      <div className="background-decorations">

        <div className="blob-1"></div>

        <div className="blob-2"></div>

      </div>

    </div>
  );
}

export default App;