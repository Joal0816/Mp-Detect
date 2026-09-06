# core/pdf_report.py - PDF report generation
"""PDF report generator for microplastic detection sessions."""
import os
from typing import List, Optional
from datetime import datetime

from core.data_models import DetectionSession, ParticleDetection


class PDFReportGenerator:
    """Generate PDF reports for microplastic detection sessions."""
    
    def __init__(self):
        self._available = False
        try:
            from reportlab.lib.pagesizes import A4
            self._available = True
        except ImportError:
            print("[PDF] reportlab not installed. PDF export disabled.")
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def generate(self, session: DetectionSession, detections: List[ParticleDetection], 
                 output_path: str) -> str:
        """Generate a PDF report for a detection session."""
        if not self._available:
            raise ImportError("reportlab is required for PDF export")
        
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, mm
        
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                               leftMargin=20*mm, rightMargin=20*mm,
                               topMargin=20*mm, bottomMargin=20*mm)
        
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Title'],
            fontSize=24, spaceAfter=20, textColor=colors.HexColor('#00E5FF')
        )
        heading_style = ParagraphStyle(
            'CustomHeading', parent=styles['Heading2'],
            fontSize=14, spaceAfter=10, textColor=colors.HexColor('#00E5FF')
        )
        
        # ── Title ───────────────────────────────────────────────────────
        story.append(Paragraph("Microplastic Detection Report", title_style))
        story.append(Spacer(1, 10))
        
        # ── Session Info ────────────────────────────────────────────────
        story.append(Paragraph("Session Information", heading_style))
        
        info_data = [
            ['Field', 'Value'],
            ['Session ID', session.session_id],
            ['Date', session.started_at],
            ['Duration', self._format_duration(session.started_at, session.ended_at)],
            ['Source', session.source_type],
            ['Model', session.model_id or 'N/A'],
            ['Confidence Threshold', f"{session.conf_threshold:.2f}"],
            ['IoU Threshold', f"{session.iou_threshold:.2f}"],
        ]
        
        if session.gps_lat is not None:
            info_data.append(['Location', f"{session.gps_lat:.4f}°, {session.gps_lon:.4f}°"])
        
        info_table = Table(info_data, colWidths=[120, 300])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16171F')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#333333')),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#0F1018')),
            ('TEXTCOLOR', (0, 1), (0, -1), colors.HexColor('#6B6B7B')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # ── Summary ─────────────────────────────────────────────────────
        story.append(Paragraph("Detection Summary", heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Particles', str(session.total_particles)],
            ['Average Confidence', f"{session.avg_confidence:.2%}"],
            ['Unique Polymer Types', str(len(session.per_class))],
            ['Unique Morphologies', str(len(session.per_morphology))],
        ]
        
        summary_table = Table(summary_data, colWidths=[150, 200])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00E5FF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#08080D')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#333333')),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # ── Polymer Breakdown ───────────────────────────────────────────
        if session.per_class:
            story.append(Paragraph("Polymer Type Breakdown", heading_style))
            
            polymer_data = [['Polymer Type', 'Count', 'Percentage']]
            for polymer, count in sorted(session.per_class.items(), 
                                         key=lambda x: x[1], reverse=True):
                pct = count / session.total_particles * 100 if session.total_particles > 0 else 0
                polymer_data.append([polymer, str(count), f"{pct:.1f}%"])
            
            polymer_table = Table(polymer_data, colWidths=[120, 80, 80])
            polymer_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16171F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#333333')),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(polymer_table)
            story.append(Spacer(1, 20))
        
        # ── Morphology Breakdown ────────────────────────────────────────
        if session.per_morphology:
            story.append(Paragraph("Morphology Breakdown", heading_style))
            
            morph_data = [['Morphology', 'Count', 'Percentage']]
            for morph, count in sorted(session.per_morphology.items(),
                                        key=lambda x: x[1], reverse=True):
                pct = count / session.total_particles * 100 if session.total_particles > 0 else 0
                morph_data.append([morph, str(count), f"{pct:.1f}%"])
            
            morph_table = Table(morph_data, colWidths=[120, 80, 80])
            morph_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16171F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#333333')),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(morph_table)
            story.append(Spacer(1, 20))
        
        # ── Size Class Breakdown ────────────────────────────────────────
        if session.per_size:
            story.append(Paragraph("Size Class Breakdown", heading_style))
            
            size_data = [['Size Class', 'Count', 'Percentage']]
            for size, count in sorted(session.per_size.items(),
                                       key=lambda x: x[1], reverse=True):
                pct = count / session.total_particles * 100 if session.total_particles > 0 else 0
                size_data.append([size, str(count), f"{pct:.1f}%"])
            
            size_table = Table(size_data, colWidths=[120, 80, 80])
            size_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16171F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#333333')),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(size_table)
            story.append(Spacer(1, 20))
        
        # ── Detailed Detections ─────────────────────────────────────────
        if detections:
            story.append(Paragraph("Detection Details", heading_style))
            
            detail_data = [['ID', 'Polymer', 'Morphology', 'Size', 'Confidence']]
            for d in detections[:100]:  # Limit to first 100
                detail_data.append([
                    str(d.particle_id),
                    d.polymer_type,
                    d.morphology,
                    d.size_class,
                    f"{d.confidence:.2%}",
                ])
            
            detail_table = Table(detail_data, colWidths=[40, 80, 80, 80, 80])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16171F')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#333333')),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(detail_table)
        
        # ── Footer ──────────────────────────────────────────────────────
        story.append(Spacer(1, 30))
        story.append(Paragraph(
            f"Generated by MP Detect v2.0.0 on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def _format_duration(self, start: str, end: Optional[str]) -> str:
        """Format duration between two ISO timestamps."""
        try:
            start_dt = datetime.fromisoformat(start)
            end_dt = datetime.fromisoformat(end) if end else datetime.now()
            delta = end_dt - start_dt
            minutes = int(delta.total_seconds() // 60)
            seconds = int(delta.total_seconds() % 60)
            if minutes > 0:
                return f"{minutes}m {seconds}s"
            return f"{seconds}s"
        except Exception:
            return "N/A"


# Singleton instance
pdf_generator = PDFReportGenerator()
