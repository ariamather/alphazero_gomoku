document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('gomoku-board');
    const ctx = canvas.getContext('2d');
    const resetButton = document.getElementById('reset-button');
    const gameStatus = document.getElementById('game-status');

    const BOARD_SIZE = 12;
    const CELL_SIZE = canvas.width / (BOARD_SIZE + 1);
    const GRID_SIZE = canvas.width;

    let board = Array(BOARD_SIZE).fill(null).map(() => Array(BOARD_SIZE).fill(0));
    let currentPlayer = 1; // 1 for black (human), -1 for white (AI)
    let gameOver = false;
    let isProcessing = false; // 防止重复请求

    function drawBoard() {
        ctx.clearRect(0, 0, GRID_SIZE, GRID_SIZE);
        ctx.fillStyle = '#f3d19c';
        ctx.fillRect(0, 0, GRID_SIZE, GRID_SIZE);

        ctx.strokeStyle = '#000';
        ctx.lineWidth = 1;

        for (let i = 0; i < BOARD_SIZE; i++) {
            // Draw vertical lines
            ctx.beginPath();
            ctx.moveTo(CELL_SIZE * (i + 1), CELL_SIZE);
            ctx.lineTo(CELL_SIZE * (i + 1), GRID_SIZE - CELL_SIZE);
            ctx.stroke();

            // Draw horizontal lines
            ctx.beginPath();
            ctx.moveTo(CELL_SIZE, CELL_SIZE * (i + 1));
            ctx.lineTo(GRID_SIZE - CELL_SIZE, CELL_SIZE * (i + 1));
            ctx.stroke();
        }
    }

    function drawPieces() {
        for (let i = 0; i < BOARD_SIZE; i++) {
            for (let j = 0; j < BOARD_SIZE; j++) {
                if (board[i][j] !== 0) {
                    const x = CELL_SIZE * (j + 1);
                    const y = CELL_SIZE * (i + 1);
                    ctx.beginPath();
                    ctx.arc(x, y, CELL_SIZE / 2.5, 0, 2 * Math.PI);
                    ctx.fillStyle = board[i][j] === 1 ? 'black' : 'white';
                    ctx.fill();
                    ctx.stroke();
                }
            }
        }
    }

    function updateBoard(newBoard) {
        // 将一维数组转换为二维数组
        for (let i = 0; i < BOARD_SIZE; i++) {
            for (let j = 0; j < BOARD_SIZE; j++) {
                board[i][j] = newBoard[i][j];
            }
        }
        drawBoard();
        drawPieces();
    }

    function updateStatus(message) {
        gameStatus.textContent = message;
    }

    async function handlePlayerMove(row, col) {
        // 防止重复点击
        if (gameOver || isProcessing || board[row][col] !== 0) {
            return;
        }

        isProcessing = true;
        
        // 立即在界面上显示玩家的棋子
        board[row][col] = currentPlayer;
        drawPieces();
        updateStatus('AI is thinking...');
        
        // 禁用点击
        canvas.style.pointerEvents = 'none';

        try {
            const response = await fetch('/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    row: row,
                    col: col
                }),
            });
            
            const data = await response.json();
            
            if (data.error) {
                // 如果出错，撤销玩家的移动
                board[row][col] = 0;
                drawPieces();
                throw new Error(data.error);
            }

            // 更新整个棋盘状态
            updateBoard(data.board);

            if (data.game_over) {
                gameOver = true;
                let winnerMsg = "Game Over! ";
                if (data.winner === 1) winnerMsg += "You win!";
                else if (data.winner === -1) winnerMsg += "AI wins!";
                else winnerMsg += "It's a draw!";
                updateStatus(winnerMsg);
            } else {
                updateStatus('Your turn (Black)');
            }

        } catch (error) {
            console.error('Error:', error);
            updateStatus('Error: ' + error.message);
            // 重新启用点击
            canvas.style.pointerEvents = 'auto';
        } finally {
            isProcessing = false;
            if (!gameOver) {
                canvas.style.pointerEvents = 'auto';
            }
        }
    }

    canvas.addEventListener('click', (event) => {
        if (gameOver || isProcessing) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        const col = Math.round(x / CELL_SIZE) - 1;
        const row = Math.round(y / CELL_SIZE) - 1;

        if (row >= 0 && row < BOARD_SIZE && col >= 0 && col < BOARD_SIZE) {
            handlePlayerMove(row, col);
        }
    });

    resetButton.addEventListener('click', async () => {
        try {
            const response = await fetch('/init', { method: 'GET' });
            const data = await response.json();
            
            // 更新棋盘
            updateBoard(data.board);
            currentPlayer = data.current_player;
            gameOver = false;
            isProcessing = false;
            updateStatus('Your turn (Black)');
            canvas.style.pointerEvents = 'auto';
        } catch (error) {
            console.error('Error:', error);
            updateStatus('Failed to reset game.');
        }
    });

        // 初始化游戏
    async function initGame() {
        try {
            console.log('Attempting to initialize game via /init');
            const response = await fetch('/init', { method: 'GET' });
            
            const data = await response.json();  // 总是尝试解析 JSON
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP error! Status: ${response.status}`);
            }
            
            console.log('Init response:', data);
            
            updateBoard(data.board);
            currentPlayer = data.current_player;
            updateStatus('Your turn (Black)');
        } catch (error) {
            console.error('Error initializing game:', error);
            updateStatus('Failed to initialize game: ' + (error.message || 'Unknown error'));
        }
    }

    // 初始化
    drawBoard();
    initGame();
});