'use client';

import { useEffect, useState, useRef } from 'react';
import axios from 'axios';

interface Point { x: number; y: number; }

export default function ChessLive() {
  const [fen, setFen] = useState('');
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [lastMove, setLastMove] = useState('');
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationPoints, setCalibrationPoints] = useState<Point[]>([]);
  const [calibrationMsg, setCalibrationMsg] = useState('');
  const imgRef = useRef<HTMLImageElement>(null);

  // Use Next.js API proxy endpoints
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get('/api/fen').then(res => {
        setFen(res.data.fen);
        setMoveHistory(res.data.move_history);
        setLastMove(res.data.last_move);
      });
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleReset = () => {
    axios.post('/api/control', { action: 'reset' });
  };

  const handleCalibrate = () => {
    setCalibrating(true);
    setCalibrationPoints([]);
    setCalibrationMsg('Click the 4 corners of the chessboard (top-left, top-right, bottom-right, bottom-left)');
  };

  const handleImgClick = (e: React.MouseEvent<HTMLImageElement, MouseEvent>) => {
    if (!calibrating || calibrationPoints.length >= 4) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * e.currentTarget.naturalWidth;
    const y = ((e.clientY - rect.top) / rect.height) * e.currentTarget.naturalHeight;
    setCalibrationPoints(points => {
      const newPoints = [...points, { x, y }];
      if (newPoints.length === 4) {
        // Send to backend
        axios.post('/api/calibrate', { corners: newPoints }).then(() => {
          setCalibrationMsg('Calibration complete!');
          setTimeout(() => {
            setCalibrating(false);
            setCalibrationMsg('');
          }, 1500);
        }).catch(() => {
          setCalibrationMsg('Calibration failed. Try again.');
        });
      }
      return newPoints;
    });
  };

  return (
    <div style={{ padding: 32 }}>
      <h1>Ultimate Chess Detector</h1>
      <div style={{ display: 'flex', gap: 32 }}>
        <div style={{ position: 'relative', width: 1800 }}>
          <img
            ref={imgRef}
            src="/api/video_feed"
            alt="Chess Video"
            style={{ width: 1800, border: '1px solid #ccc', cursor: calibrating ? 'crosshair' : 'default' }}
            onClick={handleImgClick}
          />
          {/* Draw calibration points */}
          {calibrating && calibrationPoints.map((pt, i) => (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: `${(pt.x / (imgRef.current?.naturalWidth || 640)) * 100}%`,
                top: `${(pt.y / (imgRef.current?.naturalHeight || 480)) * 100}%`,
                width: 16,
                height: 16,
                background: 'red',
                borderRadius: '50%',
                border: '2px solid white',
                transform: 'translate(-50%, -50%)',
                pointerEvents: 'none',
                zIndex: 2,
              }}
            >
              <span style={{ color: 'white', fontWeight: 'bold', fontSize: 12, position: 'absolute', left: 18, top: -8 }}>{i+1}</span>
            </div>
          ))}
          {calibrating && calibrationMsg && (
            <div style={{ position: 'absolute', top: 8, left: 8, background: 'rgba(0,0,0,0.7)', color: 'white', padding: 8, borderRadius: 4, zIndex: 3 }}>
              {calibrationMsg}
            </div>
          )}
        </div>
        <div>
          <h2>Current FEN</h2>
          <pre>{fen}</pre>
          <h2>Last Move</h2>
          <div>{lastMove}</div>
          <h2>Move History</h2>
          <ol>
            {moveHistory.map((move, i) => (
              <li key={i}>{move}</li>
            ))}
          </ol>
          <button onClick={handleReset} disabled={calibrating}>Reset Game</button>
          <button onClick={handleCalibrate} style={{ marginLeft: 8 }} disabled={calibrating}>Calibrate</button>
        </div>
      </div>
    </div>
  );
}
