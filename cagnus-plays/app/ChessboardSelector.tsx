"use client";
import React, { useRef, useEffect, useState } from "react";

const VIDEO_URL = "http://localhost:5001/video_feed";
const GAME_STATE_URL = "http://localhost:5001/game_state";

// Map FEN letters to Unicode chess symbols
const PIECE_SYMBOLS: Record<string, string> = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

export default function ChessboardSelector() {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [corners, setCorners] = useState<{ x: number; y: number }[]>([]);
  const [dimensions, setDimensions] = useState({ width: 1280, height: 720 });
  const [gameState, setGameState] = useState({
    fen: '',
    last_move: '',
    move_history: [] as string[],
    board_array: [] as string[][],
  });

  // Poll game state
  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch(GAME_STATE_URL);
        const data = await res.json();
        setGameState(data);
      } catch {
        // Ignore errors
      }
    };
    fetchState();
    const interval = setInterval(fetchState, 1000);
    return () => clearInterval(interval);
  }, []);

  // Draw markers on canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (canvas && img) {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      corners.forEach((corner, i) => {
        ctx.beginPath();
        ctx.arc(corner.x, corner.y, 16, 0, 2 * Math.PI);
        ctx.fillStyle = "red";
        ctx.fill();
        ctx.font = "32px Arial";
        ctx.fillText(`${i + 1}`, corner.x + 20, corner.y - 20);
      });
    }
  }, [corners, dimensions]);

  // Overlay detected board state
  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    const ctx = overlay.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    const board = gameState.board_array;
    if (!board || board.length !== 8) return;
    // Draw piece symbols in each square
    const squareW = overlay.width / 8;
    const squareH = overlay.height / 8;
    for (let row = 0; row < 8; row++) {
      for (let col = 0; col < 8; col++) {
        const piece = board[row][col];
        if (piece && piece !== ".") {
          const symbol = PIECE_SYMBOLS[piece] || piece;
          ctx.font = `${Math.floor(squareH * 0.7)}px Arial`;
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 4;
          ctx.strokeText(symbol, col * squareW + squareW / 2, row * squareH + squareH / 2);
          ctx.fillStyle = piece === piece.toUpperCase() ? "#222" : "#b22222";
          ctx.fillText(symbol, col * squareW + squareW / 2, row * squareH + squareH / 2);
        }
      }
    }
  }, [gameState.board_array, dimensions]);

  // Handle click
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) * img.naturalWidth) / canvas.width;
    const y = ((e.clientY - rect.top) * img.naturalHeight) / canvas.height;
    if (corners.length < 4) {
      setCorners([...corners, { x, y }]);
    }
  };

  // Send corners to backend
  const sendCorners = async () => {
    await fetch("http://localhost:5001/set_corners", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corners }),
    });
    alert("Corners sent!");
  };

  // Reset selection
  const resetCorners = () => setCorners([]);

  // On image load, update dimensions
  const handleImgLoad = () => {
    if (imgRef.current) {
      setDimensions({
        width: imgRef.current.naturalWidth,
        height: imgRef.current.naturalHeight,
      });
      if (canvasRef.current) {
        canvasRef.current.width = imgRef.current.naturalWidth;
        canvasRef.current.height = imgRef.current.naturalHeight;
      }
      if (overlayRef.current) {
        overlayRef.current.width = imgRef.current.naturalWidth;
        overlayRef.current.height = imgRef.current.naturalHeight;
      }
    }
  };

  return (
    <div style={{ position: "relative", width: "100%", height: "auto" }}>
      <img
        ref={imgRef}
        src={VIDEO_URL}
        alt="Live Chessboard"
        style={{ width: "100%", height: "auto", display: "block" }}
        onLoad={handleImgLoad}
      />
      {/* Overlay for CV board state */}
      <canvas
        ref={overlayRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
        }}
      />
      {/* Canvas for corner selection */}
      <canvas
        ref={canvasRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{
          position: "absolute",
          left: 0,
          top: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "auto",
        }}
        onClick={handleCanvasClick}
      />
      <div style={{ marginTop: 8 }}>
        <button onClick={sendCorners} disabled={corners.length !== 4}>
          Send Corners
        </button>
        <button onClick={resetCorners} style={{ marginLeft: 8 }}>
          Reset
        </button>
      </div>
      {/* Game state info */}
      <div style={{ marginTop: 24, background: '#f8f8f8', borderRadius: 8, padding: 16, width: '100%', maxWidth: 800 }}>
        <div><b>Current FEN:</b> <span style={{ fontFamily: 'monospace' }}>{gameState.fen}</span></div>
        <div><b>Last Move:</b> {gameState.last_move}</div>
        <div><b>Move History:</b>
          <ol>
            {gameState.move_history && gameState.move_history.map((move, i) => <li key={i}>{move}</li>)}
          </ol>
        </div>
      </div>
    </div>
  );
}
