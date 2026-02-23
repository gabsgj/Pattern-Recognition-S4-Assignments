import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

const GraphView = ({ transitionMatrix }) => {
    const d3Container = useRef(null);

    useEffect(() => {
        if (!transitionMatrix || transitionMatrix.length === 0 || !d3Container.current) return;

        // Clear previous
        d3.select(d3Container.current).selectAll('*').remove();

        const width = 600;
        const height = 400;
        const numStates = transitionMatrix.length;

        // Build nodes and links
        const nodes = Array.from({ length: numStates }, (_, i) => ({ id: `S${i}` }));
        const links = [];

        // Filter weak transitions for visual clarity
        const threshold = 0.05;
        for (let i = 0; i < numStates; i++) {
            for (let j = 0; j < numStates; j++) {
                if (transitionMatrix[i][j] > threshold) {
                    links.push({
                        source: `S${i}`,
                        target: `S${j}`,
                        weight: transitionMatrix[i][j],
                    });
                }
            }
        }

        const svg = d3.select(d3Container.current)
            .append('svg')
            .attr('width', '100%')
            .attr('height', '100%')
            .attr('viewBox', `0 0 ${width} ${height}`);

        // Arrow marker
        svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 22)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 8)
            .attr('markerHeight', 8)
            .attr('xoverflow', 'visible')
            .append('svg:path')
            .attr('d', 'M 0,-5 L 10 ,0 L 0,5')
            .attr('fill', '#818cf8')
            .style('stroke', 'none');

        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-400))
            .force('center', d3.forceCenter(width / 2, height / 2));

        const link = svg.append('g')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke', '#4f46e5')
            .attr('stroke-width', d => Math.max(1, d.weight * 5))
            .attr('stroke-opacity', 0.6)
            .attr('marker-end', 'url(#arrowhead)');

        const linkLabel = svg.append('g')
            .selectAll('.link-label')
            .data(links)
            .enter().append('text')
            .attr('class', 'link-label')
            .attr('fill', '#a5b4fc')
            .attr('font-size', '10px')
            .text(d => d.weight.toFixed(2));

        const node = svg.append('g')
            .selectAll('circle')
            .data(nodes)
            .enter()
            .append('g')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended)
            );

        node.append('circle')
            .attr('r', 20)
            .attr('fill', 'url(#node-gradient)')
            .attr('stroke', '#c084fc')
            .attr('stroke-width', 2);

        // Node gradient
        const gradient = svg.append("defs")
            .append("linearGradient")
            .attr("id", "node-gradient")
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "100%")
            .attr("y2", "100%");

        gradient.append("stop")
            .attr("offset", "0%")
            .style("stop-color", "#6366f1");

        gradient.append("stop")
            .attr("offset", "100%")
            .style("stop-color", "#a855f7");

        node.append('text')
            .text(d => d.id)
            .attr('text-anchor', 'middle')
            .attr('dy', 5)
            .attr('fill', '#ffffff')
            .attr('font-size', '14px')
            .attr('font-weight', 'bold');

        simulation.on('tick', () => {
            // For self-loops, we'd need paths instead of lines. We simulate it here by adding curve if source == target
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            // Adjust label position roughly to center, if self loop put it above
            linkLabel
                .attr('x', d => d.source === d.target ? d.source.x : (d.source.x + d.target.x) / 2)
                .attr('y', d => d.source === d.target ? d.source.y - 30 : (d.source.y + d.target.y) / 2 - 5);

            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });

        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }

    }, [transitionMatrix]);

    if (!transitionMatrix || transitionMatrix.length === 0) return null;

    return (
        <div className="card shadow-glass graph-card">
            <h3 className="title-gradient">State Transition Graph</h3>
            <div className="graph-container" ref={d3Container}></div>
        </div>
    );
};

export default GraphView;
