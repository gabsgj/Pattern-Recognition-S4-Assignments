from flask import Blueprint, request, jsonify
from baum_welch import baum_welch

bp = Blueprint('api', __name__)

@bp.route('/train', methods=['POST'])
def train_model():
    data = request.json
    try:
        raw_observations = data.get('observations', [])
        num_states = int(data.get('num_states', 2))
        max_iter = int(data.get('max_iter', 50))
        
        if not raw_observations:
            return jsonify({'error': 'No observations provided'}), 400
            
        # Map string observations to integers for the algorithm
        unique_obs = sorted(list(set(raw_observations)))
        obs_map = {obs: i for i, obs in enumerate(unique_obs)}
        obs_seq = [obs_map[obs] for obs in raw_observations]
        num_obs = len(unique_obs)
        
        A, B, pi, log_likelihood = baum_welch(obs_seq, num_states, num_obs, max_iter)
        
        return jsonify({
            'A': A,
            'B': B, # Map parameters back if needed, but array is fine for UI
            'pi': pi,
            'log_likelihood': log_likelihood,
            'observations_map': unique_obs # Send this back so UI knows which index corresponds to which string
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
