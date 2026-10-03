import java.io.File;
import java.util.*;
import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.profiles.*;
import org.semanticweb.owlapi.reasoner.*;
import org.semanticweb.owlapi.util.SimpleIRIMapper;

public class Check {
    static final String PAI = "https://w3id.org/prov-ai#";

    public static void main(String[] args) throws Exception {
        // args: PROV-O file, PROV-AI file, record file; the record imports PROV-AI so its properties are typed
        OWLOntologyManager m = OWLManager.createOWLOntologyManager();
        for (String iri : new String[]{"http://www.w3.org/ns/prov-o#", "http://www.w3.org/ns/prov-o", "http://www.w3.org/ns/prov#"})
            m.addIRIMapper(new SimpleIRIMapper(IRI.create(iri), IRI.create(new File(args[0]))));
        m.addIRIMapper(new SimpleIRIMapper(IRI.create("https://w3id.org/prov-ai"), IRI.create(new File(args[1]))));
        OWLOntology onto = m.loadOntologyFromOntologyDocument(new File(args[2]));
        OWLDataFactory df = m.getOWLDataFactory();
        int axioms = 0;
        for (OWLOntology o : m.getImportsClosure(onto)) axioms += o.getAxiomCount();
        System.out.println("axioms: " + axioms);
        OWLOntology vocab = m.getOntology(IRI.create("https://w3id.org/prov-ai"));
        for (OWLProfile profile : new OWLProfile[]{new OWL2DLProfile(), new OWL2RLProfile()}) {
            int ours = 0, total = 0;
            for (OWLProfileViolation v : profile.checkOntology(vocab).getViolations()) {
                total++;
                if (v.toString().contains("prov-ai")) { ours++; System.out.println("  " + v); }
            }
            System.out.println(profile.getName() + " violations: " + ours + " in PROV-AI, " + (total - ours) + " in PROV-O");
        }
        OWLReasoner r = new Reasoner.ReasonerFactory().createReasoner(onto);
        long t0 = System.currentTimeMillis();
        boolean consistent = r.isConsistent();
        System.out.println("consistent: " + consistent + " (" + (System.currentTimeMillis() - t0) + " ms)");
        if (!consistent) System.exit(1);
        r.precomputeInferences(InferenceType.CLASS_HIERARCHY);
        System.out.println("unsatisfiable classes: " + r.getUnsatisfiableClasses().getEntitiesMinusBottom().size());
        for (String p : new String[]{"dependsOn", "derivedFrom", "accountableAgent"}) {
            OWLObjectProperty prop = df.getOWLObjectProperty(IRI.create(PAI + p));
            List<String> pairs = new ArrayList<String>();
            for (OWLNamedIndividual i : onto.getIndividualsInSignature(false))
                for (OWLNamedIndividual j : r.getObjectPropertyValues(i, prop).getFlattened())
                    pairs.add(i.getIRI() + " " + j.getIRI());
            Collections.sort(pairs);
            for (String s : pairs) System.out.println(p + " " + s);
        }
    }
}
