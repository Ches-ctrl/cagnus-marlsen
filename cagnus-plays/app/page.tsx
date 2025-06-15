'use client';

import { useState } from 'react';
import ChessLive from './ChessLive';
import Chessboard from './Chessboard';

export default function Page() {
  // Calibration state lifted up
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationPoints, setCalibrationPoints] = useState<{x: number, y: number}[]>([]);
  const [calibrationMsg, setCalibrationMsg] = useState('');
  const [fen, setFen] = useState('');

  // Handler to start calibration
  const handleCalibrate = () => {
    setCalibrating(true);
    setCalibrationPoints([]);
    setCalibrationMsg('Click the 4 corners of the chessboard (top-left, top-right, bottom-right, bottom-left)');
  };

  // Handler for image click
  const handleImgClick = (x: number, y: number) => {
    if (!calibrating || calibrationPoints.length >= 4) return;
    const newPoints = [...calibrationPoints, { x, y }];
    setCalibrationPoints(newPoints);
    if (newPoints.length === 4) {
      // Send to backend
      fetch('/api/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ corners: newPoints })
      })
        .then(() => {
          setCalibrationMsg('Calibration complete!');
          setTimeout(() => {
            setCalibrating(false);
            setCalibrationMsg('');
          }, 1500);
        })
        .catch(() => {
          setCalibrationMsg('Calibration failed. Try again.');
        });
    }
  };

  // Handler to update FEN from ChessLive
  const handleFenUpdate = (newFen: string) => {
    setFen(newFen);
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f7f9fb', fontFamily: 'Inter, Arial, sans-serif', position: 'relative' }}>
      {/* Norwegian flag in top right */}
      <a href="https://en.wikipedia.org/wiki/Magnus_Carlsen" target="_blank" rel="noopener noreferrer">
        <img
          src="https://upload.wikimedia.org/wikipedia/commons/d/d9/Flag_of_Norway.svg"
          alt="Norwegian Flag"
          style={{ position: 'fixed', top: 24, right: 32, width: 64, height: 42, boxShadow: '0 2px 12px rgba(0,0,0,0.18)', borderRadius: 6, border: '2px solid #fff', zIndex: 10 }}
        />
      </a>
      {/* Title */}
      <h1
        style={{
          fontSize: '2.4rem',
          fontWeight: 900,
          letterSpacing: '0.04em',
          margin: '0 auto',
          marginTop: 32,
          marginBottom: 32,
          textAlign: 'center',
          fontFamily: 'Inter, Arial, sans-serif',
          WebkitBackgroundClip: 'text',
          textShadow: '0 2px 8px #00286822, 0 1px 0 #fff',
        }}
      >
        Cangus Marlsen Chess Championship 2025
      </h1>
      {/* Main content area */}
      <div
        style={{
          display: 'flex', flexDirection: 'row', alignItems: 'flex-start', justifyContent: 'center',
          width: '100vw', minHeight: '80vh',
          gap: 48,
        }}
      >
        {/* Left column: details/controls */}
        <div
          style={{
            minWidth: 440,
            maxWidth: 540,
            background: '#fff',
            borderRadius: 16,
            boxShadow: '0 2px 16px rgba(0,0,0,0.07)',
            padding: 32,
            marginTop: 16,
            marginBottom: 32,
            display: 'flex', flexDirection: 'column', alignItems: 'flex-start',
          }}
        >
          <ChessLive
            mode="details"
            calibrating={calibrating}
            calibrationMsg={calibrationMsg}
            onCalibrate={handleCalibrate}
            onFenUpdate={handleFenUpdate}
          />
          <div style={{ marginTop: 32, width: '100%', display: 'flex', justifyContent: 'center' }}>
            <Chessboard fen={fen} />
          </div>
        </div>
        {/* Center: video */}
        <div
          style={{
            background: '#fff',
            borderRadius: 16,
            boxShadow: '0 2px 16px rgba(0,0,0,0.07)',
            padding: 24,
            marginTop: 16,
            marginBottom: 32,
            display: 'flex', flexDirection: 'column', alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <ChessLive
            mode="video"
            calibrating={calibrating}
            calibrationPoints={calibrationPoints}
            calibrationMsg={calibrationMsg}
            onImgClick={handleImgClick}
            onFenUpdate={handleFenUpdate}
          />
        </div>
      </div>
    </div>
  );
}
