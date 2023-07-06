import React from 'react'
import Header2 from './Header2'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faGithub, faLinkedin, faFacebook  } from '@fortawesome/free-brands-svg-icons'
import { faEnvelope } from '@fortawesome/free-regular-svg-icons'
import { faCode, faDatabase } from '@fortawesome/free-solid-svg-icons'
import { Link } from 'react-router-dom'
import Header3 from './Header3'

function Footer() {
    return (
        <footer className='bg-dark text-white mt-auto'>
            <div className='container p-4 h-100'>
                <section className='d-flex justify-content-around mb-4'>
                        <div className='' style={{margin: "0 auto 0 0"}}>
                            <Header2 className=''>Get Repository & More!</Header2>
                        </div>
                        <div className='' style={{margin: "auto 0 0 0"}}>
                            <Link to="https://github.com/Messerschmidts101/Churn-Prediction-V1" className="btn btn-outline-light btn-floating m-1">
                                <FontAwesomeIcon icon={faGithub} />
                            </Link>
                            <Link to="https://www.linkedin.com/in/matthew-olaguer-683885245/" className="btn btn-outline-light btn-floating m-1">
                                <FontAwesomeIcon icon={faLinkedin} />
                            </Link>
                            <Link to="https://www.facebook.com/Flint.Oldfield/" className="btn btn-outline-light btn-floating m-1">
                                <FontAwesomeIcon icon={faFacebook} />
                            </Link>
                    </div>
                </section>
                <section className='mb-4 text-center'>
                    <p>
                        At TelCo Generic, we specialize in customer churn analysis, empowering businesses to reduce churn rates and enhance customer retention. Through advanced data analytics and predictive modeling, we uncover valuable insights and provide actionable strategies to optimize customer loyalty. Partner with us to unlock the potential of your customer base and drive sustainable growth through data-driven retention solutions.
                    </p>
                </section>
                <section className='container'>
                    <div className='row'>
                        <div className='col'>
                            <Header2>
                                <FontAwesomeIcon icon={faEnvelope} /> Contacts
                            </Header2>
                            <p className='asparagus'>A. Mabini Campus, Anonas Street, Sta. Mesa Manila, Philippines 1016</p>
                            <p className='text-info'><a className='mail' href='mailto:inquire@pup.edu.ph'>inquire@pup.edu.ph</a></p>
                            <p className='text-info'><a className='mail' href='tel:(632)8713-1505'>(632) 8713-1505</a></p>
                        </div>
                        <div className='col-md-4 col-sm-9 border-start ps-5'>
                            <Header3><FontAwesomeIcon icon={faCode} /> WebDev Team</Header3>
                            <p><a className='mail' href='mailto:jcglintag@iskolarngbayan.pup.edu.ph'>Lintag, Juan Carlo</a></p>
                            <p><a className='mail' href='mailto:mcolaguer@iskolarngbayan.pup.edu.ph'>Olaguer, Matthew</a></p>
                            <p><a className='mail' href='mailto:jaapaano@iskolarngbayan.pup.edu.ph'>Paano, Julius Angelo</a></p>
                        </div>
                        <div className='col-md-4 col-sm-9'>
                            <Header3><FontAwesomeIcon icon={faDatabase} /> Data Science Team </Header3>
                            <p><a className='mail' href='mailto:caalarcon@iskolarngbayan.pup.edu.ph'>Alarcon, Chastin</a></p>
                            <p><a className='mail' href='mailto:acddavid@iskolarngbayan.pup.edu.ph'>David, Adrien Christian</a></p>
                            <p><a className='mail' href='bkmferrer@iskolarngbayan.pup.edu.ph'>Ferrer, Bryan Kristoffer</a></p>
                            <p><a className='mail' href='mailto:esssenorin@iskolarngbayan.pup.edu.ph'>Señorin, Ereka Sheen</a></p>
                        </div>
                    </div>
                </section>
            </div>
            <div className='bg-dark-green text-center p-4'>
                © 2021 Copyright: <a className='text-reset fw-bold' href='der-schneeprinz.github.io'>Der-Schneeprinz</a>
            </div>
        </footer>
    )
}

export default Footer