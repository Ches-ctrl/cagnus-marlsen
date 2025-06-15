'use client';

import { useEffect, useState } from 'react';
import axios from 'axios';

export default function ChessLive() {
  const [fen, setFen] = useState('');
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [lastMove, setLastMove] = useState('');

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
    axios.post('/api/control', { action: 'calibrate' });
  };

  return (
    <div style={{ padding: 32 }}>
      <h1>Ultimate Chess Detector</h1>
      <div style={{ display: 'flex', gap: 32 }}>
        <div>
          <img
            src="/api/video_feed"
            alt="Chess Video"
            style={{ width: 640, border: '1px solid #ccc' }}
          />
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
          <button onClick={handleReset}>Reset Game</button>
          <button onClick={handleCalibrate} style={{ marginLeft: 8 }}>Calibrate</button>
        </div>
      </div>
    </div>
  );
}
