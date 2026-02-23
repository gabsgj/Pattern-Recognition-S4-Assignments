import React, { useMemo, useRef, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import './StateGraph.css';

const StateGraph = ({ states, A }) => {
    const fgRef = useRef();

    // Re-center graph automatically
    useEffect(() => {
        if (fgRef.current) {
            fgRef.current.d3Force('charge').strength(-400); // Spread nodes
            fgRef.current.zoomToFit(400, 50); // Zoom animation
        }
    }, [A]);

    const graphData = useMemo(() => {
        const nodes = states.map((state, i) => ({
            id: i,
            name: state,
            val: 20 // size
        }));

        const links = [];
        for (let i = 0; i < A.length; i++) {
            for (let j = 0; j < A[i].length; j++) {
                const weight = A[i][j];
                if (weight > 0.01) { // Only show significant links
                    links.push({
                        source: i,
                        target: j,
                        weight: weight,
                        label: weight.toFixed(3)
                    });
                }
            }
        }

        return { nodes, links };
    }, [states, A]);

    return (
        <div className="state-graph-container glass-panel">
            <h3 className="section-title">State Transitions Graph</h3>
            <div className="graph-wrapper">
                <ForceGraph2D
                    ref={fgRef}
                    width={600}
                    height={400}
                    graphData={graphData}
                    nodeLabel="name"
                    nodeColor={() => '#8b5cf6'}
                    nodeRelSize={8}
                    linkColor={() => 'rgba(139, 92, 246, 0.4)'}
                    linkWidth={link => Math.max(1, link.weight * 5)}
                    linkDirectionalArrowLength={3.5}
                    linkDirectionalArrowRelPos={1}
                    linkCurvature={0.2} // Curved edges to see loops
                    nodeCanvasObject={(node, ctx, globalScale) => {
                        const label = node.name;
                        const fontSize = 14 / globalScale;
                        ctx.font = `${fontSize}px Outfit, sans-serif`;
                        const textWidth = ctx.measureText(label).width;
                        const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

                        ctx.fillStyle = '#1e293b';
                        ctx.beginPath();
                        ctx.arc(node.x, node.y, node.val / globalScale, 0, 2 * Math.PI, false);
                        ctx.fill();
                        ctx.lineWidth = 2 / globalScale;
                        ctx.strokeStyle = '#8b5cf6';
                        ctx.stroke();

                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillStyle = '#fff';
                        ctx.fillText(label, node.x, node.y);
                    }}
                    linkCanvasObjectMode={() => 'after'}
                    linkCanvasObject={(link, ctx, globalScale) => {
                        const MAX_FONT_SIZE = 12;
                        const LABEL_NODE_MARGIN = 20;

                        const start = link.source;
                        const end = link.target;
                        if (!start || !end || typeof start.x !== 'number') return;

                        const isLoop = start.id === end.id;

                        // Draw link label
                        const label = link.label;
                        const fontSize = MAX_FONT_SIZE / globalScale;
                        ctx.font = `500 ${fontSize}px Inter, sans-serif`;

                        let x, y;
                        if (isLoop) {
                            x = start.x;
                            y = start.y - 25; // Offset loop label upwards
                        } else {
                            // Midpoint calculation for curved edge
                            // Using a simple midpoint approximation for Bezier curvature of 0.2
                            x = start.x + (end.x - start.x) / 2;
                            y = start.y + (end.y - start.y) / 2;
                            // Add slight offset for curvature
                            const len = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
                            if (len > 0) {
                                x += ((end.y - start.y) / len) * (len * 0.2);
                                y -= ((end.x - start.x) / len) * (len * 0.2);
                            }
                        }

                        ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
                        const padding = 2;
                        const tw = ctx.measureText(label).width;
                        const th = fontSize;
                        ctx.fillRect(x - tw / 2 - padding, y - th / 2 - padding, tw + padding * 2, th + padding * 2);

                        ctx.fillStyle = '#fbbf24'; // Premium gold color for transitions
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(label, x, y);
                    }}
                />
            </div>
        </div>
    );
};

export default StateGraph;
