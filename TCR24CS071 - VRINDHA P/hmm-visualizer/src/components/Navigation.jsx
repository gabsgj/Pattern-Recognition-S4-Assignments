import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BookOpen, LineChart, Network, Settings } from 'lucide-react';
import './Navigation.css';

const TABS = [
    { id: 'playground', label: 'Playground', icon: Settings },
    { id: 'graph', label: 'Transitions', icon: Network },
    { id: 'optimization', label: 'Optimization', icon: LineChart },
    { id: 'deepdive', label: 'Deep Dive', icon: BookOpen },
];

const Navigation = ({ activeTab, setActiveTab }) => {
    return (
        <nav className="site-nav">
            <div className="nav-container">
                <ul className="nav-list">
                    {TABS.map((tab) => {
                        const Icon = tab.icon;
                        const isActive = activeTab === tab.id;

                        return (
                            <li key={tab.id} className="nav-item">
                                <button
                                    onClick={() => setActiveTab(tab.id)}
                                    className={`nav-button ${isActive ? 'active' : ''}`}
                                >
                                    <Icon size={18} className="nav-icon" />
                                    <span className="nav-label">{tab.label}</span>

                                    {isActive && (
                                        <motion.div
                                            layoutId="active-nav-indicator"
                                            className="nav-active-bg"
                                            initial={false}
                                            transition={{
                                                type: "spring",
                                                stiffness: 500,
                                                damping: 35,
                                            }}
                                        />
                                    )}
                                </button>
                            </li>
                        );
                    })}
                </ul>
            </div>
        </nav>
    );
};

export default Navigation;
